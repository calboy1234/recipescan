"""
recipe_detector.py

Score-based heuristic recipe detector.

Each signal contributes points toward a total score (0.0–1.0).
A score >= RECIPE_THRESHOLD is classified as a recipe.

Signals
-------
- Recipe keywords (title/section words)     → up to 0.35
- Measurement units                         → up to 0.35
- Fraction / quantity patterns              → up to 0.20
- Ingredient list structure (bullets/lines) → up to 0.10

Note: Numbered step lists are intentionally NOT scored — many valid recipes
(especially handwritten cards or short/simple ones) have no numbered steps,
and this signal was causing false negatives.
"""

import re

# ── Tuneable threshold ────────────────────────────────────────────────────────
RECIPE_THRESHOLD = 0.40   # score at or above this → is_recipe = True

# ── Signal word lists ─────────────────────────────────────────────────────────
RECIPE_KEYWORDS = [
    "ingredients", "ingredient", "directions", "instructions", "method",
    "preparation", "preheat", "serves", "servings", "yield", "yields",
    "prep time", "cook time", "bake", "baking", "recipe", "makes",
    "refrigerate", "marinate", "garnish", "season to taste", "stir",
    "simmer", "boil", "whisk", "fold in", "mix", "combine",
]

MEASUREMENT_UNITS = [
    r"\bcup s?\b", r"\bc\.\b",
    r"\btablespoon s?\b", r"\btbsp\.?\b", r"\bT\b",
    r"\bteaspoon s?\b",  r"\btsp\.?\b",
    r"\bounce s?\b",     r"\boz\.?\b",
    r"\bpound s?\b",     r"\blb s?\.?\b",
    r"\bgram s?\b",      r"\bg\b",
    r"\bkilogram s?\b",  r"\bkg\b",
    r"\bmilliliter s?\b",r"\bml\b",
    r"\bliter s?\b",     r"\bl\b",
    r"\bpinch\b",        r"\bdash\b",
    r"\bslice s?\b",     r"\bclove s?\b",
    r"\bcan s?\b",       r"\bpackage s?\b",
    r"\bstick s?\b",     r"\bsprig s?\b",
]

# Compiled once at import time
_KW_PATTERN   = re.compile(
    r'\b(' + '|'.join(re.escape(k) for k in RECIPE_KEYWORDS) + r')\b',
    re.IGNORECASE
)
_UNIT_PATTERN = re.compile(
    '(' + '|'.join(MEASUREMENT_UNITS) + ')',
    re.IGNORECASE | re.VERBOSE
)
_FRACTION_PATTERN  = re.compile(r'\b\d+\s*/\s*\d+\b|\b\d+\.\d+\b')
_BULLET_LINE_PATTERN = re.compile(r'^\s*[-•*]\s+\w', re.MULTILINE)

# A "quantity + word" pattern: "2 eggs", "3 cloves garlic"
_QTY_WORD_PATTERN  = re.compile(r'\b\d+\s+[a-zA-Z]')


def _clamp(value, lo=0.0, hi=1.0):
    return max(lo, min(hi, value))


def score_text(text: str) -> dict:
    """
    Analyse `text` and return a dict with:
        score       float   0.0–1.0
        is_recipe   bool
        signals     dict    breakdown of each sub-score
    """
    if not text or not text.strip():
        return {"score": 0.0, "is_recipe": False, "signals": {}}

    lines = text.splitlines()
    word_count = len(text.split())

    # ── 1. Recipe keyword hits (max 0.35) ────────────────────────────────────
    kw_hits = len(_KW_PATTERN.findall(text))
    # 1 hit = 0.12, 2 = 0.24, 3+ = 0.35
    kw_score = _clamp(kw_hits * 0.12, 0, 0.35)

    # ── 2. Measurement unit density (max 0.35) ────────────────────────────────
    unit_hits = len(_UNIT_PATTERN.findall(text))
    # Normalise by word count to avoid rewarding long non-recipes
    unit_density = unit_hits / max(word_count, 1)
    unit_score = _clamp(unit_density * 35, 0, 0.35)

    # ── 3. Fractions / quantities (max 0.20) ──────────────────────────────────
    frac_hits = len(_FRACTION_PATTERN.findall(text))
    qty_hits  = len(_QTY_WORD_PATTERN.findall(text))
    frac_score = _clamp((frac_hits + qty_hits) * 0.04, 0, 0.20)

    # ── 4. Bullet / short-line list (max 0.10) ───────────────────────────────
    bullet_hits     = len(_BULLET_LINE_PATTERN.findall(text))
    short_lines     = sum(1 for l in lines if 3 < len(l.strip()) < 60)
    ingredient_feel = _clamp((bullet_hits * 0.05) + (short_lines / max(len(lines), 1) * 0.10), 0, 0.10)

    total = kw_score + unit_score + frac_score + ingredient_feel

    signals = {
        "keyword_hits":   kw_hits,
        "keyword_score":  round(kw_score, 3),
        "unit_hits":      unit_hits,
        "unit_score":     round(unit_score, 3),
        "fraction_hits":  frac_hits + qty_hits,
        "fraction_score": round(frac_score, 3),
        "bullet_hits":    bullet_hits,
        "list_score":     round(ingredient_feel, 3),
    }

    return {
        "score":     round(_clamp(total), 4),
        "is_recipe": total >= RECIPE_THRESHOLD,
        "signals":   signals,
    }


def extract_title(text: str) -> str | None:
    """
    Best-effort title extraction from OCR text.

    Strategy:
    1. Reject lines that are mostly noise (symbols, single chars, low alnum ratio)
    2. Prefer ALL-CAPS lines — recipe card titles are very often all-caps
    3. Fall back to the first clean mixed-case line
    4. Return None if nothing clean is found
    """
    # A line is "clean" if ≥60% of its characters are alphanumeric or spaces
    def alnum_ratio(s):
        if not s:
            return 0
        alnum = sum(1 for c in s if c.isalnum() or c == ' ')
        return alnum / len(s)

    # A word is "real" if it has ≥2 letters
    def real_word_count(s):
        return sum(1 for w in s.split() if sum(c.isalpha() for c in w) >= 2)

    candidates_allcaps = []
    candidates_normal  = []

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if len(line) > 80:
            continue                        # too long to be a title
        if len(line) < 4:
            continue                        # too short, probably noise
        if alnum_ratio(line) < 0.60:
            continue                        # mostly symbols/garbage
        if real_word_count(line) < 2:
            continue                        # fewer than 2 real words

        # Strip any leading noise characters (Z, * etc.) before a capital word
        cleaned = re.sub(r'^[\W_]+', '', line).strip()
        if not cleaned or real_word_count(cleaned) < 2:
            continue

        if cleaned == cleaned.upper() and any(c.isalpha() for c in cleaned):
            candidates_allcaps.append(cleaned)
        else:
            candidates_normal.append(cleaned)

    if candidates_allcaps:
        return candidates_allcaps[0]
    if candidates_normal:
        return candidates_normal[0]
    return None


if __name__ == "__main__":
    # Quick smoke-test
    sample = """
    Chocolate Chip Cookies

    Ingredients:
    - 2 1/4 cups all-purpose flour
    - 1 tsp baking soda
    - 1 tsp salt
    - 1 cup (2 sticks) butter, softened
    - 3/4 cup granulated sugar

    Directions:
    1. Preheat oven to 375°F.
    2. Combine flour, baking soda and salt in a bowl.
    3. Mix butter and sugars until creamy.
    4. Bake for 9–11 minutes.
    """
    result = score_text(sample)
    print(f"Score:     {result['score']}")
    print(f"Is recipe: {result['is_recipe']}")
    print(f"Signals:   {result['signals']}")
    print(f"Title:     {extract_title(sample)}")
