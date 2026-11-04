import re
import string
import unicodedata
from fractions import Fraction


# ---------- GSM8K ----------
def gsm8k_gold(answer_text):
    """Gold answer is the number after '####'."""
    return answer_text.split("####")[-1].strip().replace(",", "")


def last_boxed(text):
    """Content of the last \\boxed{...}, handling nested braces."""
    start = text.rfind("\\boxed{")
    if start == -1:
        return None
    i = start + len("\\boxed{")
    depth = 1
    out = []
    while i < len(text) and depth > 0:
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                break
        out.append(c)
        i += 1
    return "".join(out) if depth == 0 else None


def to_number(s):
    """Turn '1,234', '$12.50', '50%', '\\frac{11}{2}', '11/2' into a float. None if impossible."""
    if s is None:
        return None
    s = s.strip()
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    m = re.search(r"\\frac\{\s*(-?[\d.]+)\s*\}\{\s*(-?[\d.]+)\s*\}", s)
    if m:
        try:
            return float(Fraction(m.group(1)) / Fraction(m.group(2)))
        except (ValueError, ZeroDivisionError):
            return None
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    s = s.replace(",", "").replace("$", "").replace("%", "").replace("\\", "")
    m = re.search(r"-?\d+(?:\.\d+)?\s*/\s*-?\d+(?:\.\d+)?", s)
    if m:
        try:
            a, b = m.group(0).split("/")
            return float(Fraction(a.strip()) / Fraction(b.strip()))
        except (ValueError, ZeroDivisionError):
            return None
    m = re.search(r"-?\d+(?:\.\d+)?", s)
    return float(m.group(0)) if m else None


def gsm8k_pred(output):
    """Prediction = last \\boxed{} if present, otherwise the last number in the text."""
    boxed = last_boxed(output)
    if boxed is not None:
        return boxed, "boxed"
    nums = re.findall(r"-?\d[\d,]*(?:\.\d+)?", output)
    return (nums[-1] if nums else None), "last_number"


def gsm8k_correct(pred, gold):
    p, g = to_number(pred), to_number(gold)
    return p is not None and g is not None and abs(p - g) < 1e-4


# ---------- TriviaQA ----------
def normalize_answer(s):
    """Standard SQuAD/TriviaQA normalization: lowercase, no accents, no punctuation, no articles."""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))   # Río -> Rio
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def trivia_pred(output):
    """Keep the first line and drop a leading 'Answer:'."""
    line = output.strip().split("\n")[0].strip()
    line = re.sub(r"^(final\s+)?answer\s*:\s*", "", line, flags=re.I)
    return line


def trivia_scores(pred, aliases):
    """em: normalized prediction equals an alias.
    contains: an alias appears as whole words inside a short prediction (<= 10 words)."""
    p = normalize_answer(pred)
    norm_aliases = {normalize_answer(a) for a in aliases if a.strip()}
    em = p in norm_aliases
    contains = len(p.split()) <= 10 and any(
        a and re.search(r"(^| )" + re.escape(a) + r"( |$)", p) for a in norm_aliases
    )
    return em, contains
