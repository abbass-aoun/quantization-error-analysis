# RTN and GPTQ in plain PyTorch, following the original GPTQ code
# (IST-DASLab/gptq: quant.py, gptq.py, opt.py). Both methods use the SAME
# Quantizer, so they share exactly the same grid definition.
import copy
import math
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# 1. The grid (asymmetric min-max, as in Section 2.1 of the survey)
# ---------------------------------------------------------------------------
def quantize(x, scale, zero, maxq):
    """q = clip(round(x/s) + z, 0, 2^b - 1);  w_hat = s * (q - z)"""
    q = torch.clamp(torch.round(x / scale) + zero, 0, maxq)
    return scale * (q - zero)


class Quantizer:
    def __init__(self, bits, sym=False):
        self.maxq = 2 ** bits - 1
        self.sym = sym

    def find_params(self, w):
        """One scale and zero-point per row of w (w = one group of columns)."""
        xmin = torch.minimum(w.min(dim=1).values, torch.zeros(1, device=w.device))
        xmax = torch.maximum(w.max(dim=1).values, torch.zeros(1, device=w.device))
        if self.sym:
            xmax = torch.maximum(xmin.abs(), xmax)
            xmin = -xmax
        both_zero = (xmin == 0) & (xmax == 0)
        xmin[both_zero], xmax[both_zero] = -1, 1
        self.scale = ((xmax - xmin) / self.maxq).unsqueeze(1)
        if self.sym:
            self.zero = torch.full_like(self.scale, (self.maxq + 1) / 2)
        else:
            self.zero = torch.round(-xmin / self.scale.squeeze(1)).unsqueeze(1)

    def __call__(self, x):
        return quantize(x, self.scale, self.zero, self.maxq)


# ---------------------------------------------------------------------------
# 2. RTN: round each group independently, no data
# ---------------------------------------------------------------------------
@torch.no_grad()
def rtn_linear_(layer, bits, groupsize, sym=False):
    W = layer.weight.data.float()
    cols = W.shape[1]
    g = cols if groupsize == -1 else groupsize
    qz = Quantizer(bits, sym)
    Q = torch.empty_like(W)
    for c in range(0, cols, g):
        qz.find_params(W[:, c:c + g])
        Q[:, c:c + g] = qz(W[:, c:c + g])
    layer.weight.data = Q.to(layer.weight.dtype)


# ---------------------------------------------------------------------------
# 3. GPTQ for one linear layer (Algorithm 1 of the paper)
# ---------------------------------------------------------------------------
class GPTQ:
    def __init__(self, layer):
        self.layer = layer
        cols = layer.weight.shape[1]
        self.H = torch.zeros((cols, cols), device=layer.weight.device, dtype=torch.float32)
        self.nsamples = 0

    def add_batch(self, inp):
        """Accumulate H = 2 X X^T / n from the inputs this layer receives."""
        inp = inp.reshape(-1, inp.shape[-1]).t().float()        # (cols, tokens)
        self.H *= self.nsamples / (self.nsamples + 1)
        self.nsamples += 1
        inp = math.sqrt(2 / self.nsamples) * inp
        self.H += inp @ inp.t()

    @torch.no_grad()
    def quantize_(self, bits, groupsize=128, sym=False, percdamp=0.01, blocksize=128,
                  act_order=False, static_groups=False):
        """act_order: quantize columns with the largest H diagonal first ("desc_act").
        static_groups: each group's scale/zero-point is fixed from the ORIGINAL weights,
        so GPTQ uses exactly the same grid as RTN (only rounding decisions differ)."""
        W = self.layer.weight.data.clone().float()
        rows, cols = W.shape
        qz = Quantizer(bits, sym)
        if groupsize == -1:
            qz.find_params(W)
        H = self.H
        del self.H
        dead = torch.diag(H) == 0                    # inputs that are always 0
        H[dead, dead] = 1
        W[:, dead] = 0
        groups = None
        if groupsize != -1 and (static_groups or act_order):
            groups = []
            for c in range(0, cols, groupsize):
                g = Quantizer(bits, sym)
                g.find_params(W[:, c:c + groupsize])
                groups.append(g)
        perm = None
        if act_order:
            perm = torch.argsort(torch.diag(H), descending=True)
            W = W[:, perm]
            H = H[perm][:, perm]
            invperm = torch.argsort(perm)
        damp = percdamp * torch.mean(torch.diag(H))  # 1% dampening
        H += damp * torch.eye(cols, device=H.device)
        # Cholesky form of H^-1 (upper triangular), as in the paper
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        Hinv = torch.linalg.cholesky(H, upper=True)

        Q = torch.zeros_like(W)
        losses = torch.zeros(rows, device=W.device)
        for i1 in range(0, cols, blocksize):          # lazy batch updates
            i2 = min(i1 + blocksize, cols)
            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]
            for i in range(i2 - i1):                  # fixed column order
                if groups is not None:
                    col = int(perm[i1 + i]) if perm is not None else i1 + i
                    qz = groups[col // groupsize]
                elif groupsize != -1 and (i1 + i) % groupsize == 0:
                    qz.find_params(W[:, i1 + i:i1 + i + groupsize])
                w = W1[:, i]
                d = Hinv1[i, i]
                q = qz(w.unsqueeze(1)).flatten()
                Q1[:, i] = q
                losses += (w - q) ** 2 / d ** 2 / 2
                err = (w - q) / d
                W1[:, i:] -= err.unsqueeze(1) @ Hinv1[i, i:].unsqueeze(0)
                Err1[:, i] = err
            Q[:, i1:i2] = Q1
            W[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]
        if perm is not None:
            Q = Q[:, invperm]
        self.layer.weight.data = Q.to(self.layer.weight.dtype)
        return losses.sum().item()


# ---------------------------------------------------------------------------
# 4. Whole-model helpers
# ---------------------------------------------------------------------------
def get_layers(model):
    """Transformer blocks for Qwen/Llama-style and OPT-style models."""
    inner = model.model
    if hasattr(inner, "layers"):
        return inner.layers
    if hasattr(inner, "decoder") and hasattr(inner.decoder, "layers"):
        return inner.decoder.layers
    raise ValueError("Unknown architecture")


def block_linears(block):
    return {n: m for n, m in block.named_modules() if isinstance(m, nn.Linear)}


@torch.no_grad()
def rtn_model_(model, bits=4, groupsize=128, sym=False):
    for block in get_layers(model):
        for m in block_linears(block).values():
            rtn_linear_(m, bits, groupsize, sym)


class _Catcher(nn.Module):
    """Stands in for block 0 to record its inputs, then stops the forward pass."""
    def __init__(self, m, store):
        super().__init__()
        self.m = m
        self.store = store

    def __getattr__(self, name):          # pass through attributes like attention_type
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.m, name)

    def forward(self, *args, **kwargs):
        hs = args[0] if args else kwargs.pop("hidden_states")
        self.store["inps"].append(hs)
        if self.store["kwargs"] is None:
            self.store["kwargs"] = kwargs
        raise ValueError("captured")


def _first(out):
    return out[0] if isinstance(out, (tuple, list)) else out


# Order used by "true sequential" GPTQ (Llama/Qwen-style blocks)
TRUE_SEQ = [["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
            ["self_attn.o_proj"], ["mlp.gate_proj", "mlp.up_proj"], ["mlp.down_proj"]]


@torch.no_grad()
def gptq_model_(model, calib, bits=4, groupsize=128, sym=False,
                percdamp=0.01, blocksize=128, verbose=True,
                act_order=False, static_groups=False, true_sequential=False):
    """Quantize block by block. Each block is calibrated on the outputs of the
    blocks before it that are ALREADY quantized (as in the GPTQ paper)."""
    layers = get_layers(model)
    dev = next(model.parameters()).device
    store = {"inps": [], "kwargs": None}
    layers[0] = _Catcher(layers[0], store)
    for x in calib:
        try:
            model(x.to(dev), use_cache=False)
        except ValueError:
            pass
    layers[0] = layers[0].m
    inps = torch.cat(store["inps"], dim=0)
    kw = store["kwargs"]
    outs = torch.zeros_like(inps)

    for li, block in enumerate(layers):
        lins = block_linears(block)
        names = list(lins)
        if true_sequential and all(n in lins for grp in TRUE_SEQ for n in grp) \
                and len(names) == sum(len(g) for g in TRUE_SEQ):
            steps = TRUE_SEQ     # quantize q/k/v, then o, then gate/up, then down
        else:
            steps = [names]      # all layers of the block at once (original GPTQ code)
        err = {}
        for step in steps:
            gptq = {n: GPTQ(lins[n]) for n in step}
            hooks = [lins[n].register_forward_hook(
                        lambda mod, inp, out, n=n: gptq[n].add_batch(inp[0].data))
                     for n in step]
            for j in range(inps.shape[0]):
                block(inps[j:j + 1], **kw)
            for h in hooks:
                h.remove()
            for n in step:
                err[n] = gptq[n].quantize_(bits, groupsize, sym, percdamp, blocksize,
                                           act_order, static_groups)
                del gptq[n]
        for j in range(inps.shape[0]):           # outputs of the QUANTIZED block
            outs[j] = _first(block(inps[j:j + 1], **kw))[0]
        inps, outs = outs, inps
        if verbose:
            print(f"  block {li:2d} done | total error {sum(err.values()):.1f}")
        torch.cuda.empty_cache() if torch.cuda.is_available() else None


# ---------------------------------------------------------------------------
# 5. Check: every (row, group) uses at most 2^bits distinct values
# ---------------------------------------------------------------------------
@torch.no_grad()
def check_grid(model, bits, groupsize):
    worst = 0
    for block in get_layers(model):
        for m in block_linears(block).values():
            W = m.weight.data.float()
            g = W.shape[1] if groupsize == -1 else groupsize
            for c in range(0, W.shape[1], g):
                grp = W[:, c:c + g]
                worst = max(worst, max(len(torch.unique(r)) for r in grp[:8]))
    return worst
