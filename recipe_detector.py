import re
import unicodedata

# ── Tuneable threshold ────────────────────────────────────────────────────────
RECIPE_THRESHOLD = 0.60 

# ── Signal patterns ───────────────────────────────────────────────────────────

RECIPE_KEYWORDS = [
    "ingredients", "ingredient", "preheat", "serves", "servings",
    "prep time", "cook time", "bake", "baking", "recipe", "makes",
    "refrigerate", "marinate", "garnish", "stir", "simmer", "boil", 
    "whisk", "fold in", "mix", "combine", "sauté", "chop", "mince", 
    "dice", "puree", "drain", "grease", "sprinkle", "knead", "dissolve", 
    "skillet", "saucepan", "oven", "rack", "instructions", "directions", "pan",
    "pot", "baking sheet", "cookie sheet", "casserole dish", "slow cooker", "instant pot",
    "grill", "broil", "roast", "steam", "fry", "deep fry", "air fry",
    "blend", "stand mixer", "hand mixer", "mixing bowl", "blender"
]

COMMON_INGREDIENTS = [
    # Base pantry
    "salt", "sea salt", "kosher salt", "black pepper", "white pepper",
    "sugar", "granulated sugar", "brown sugar", "powdered sugar", "confectioners sugar",
    "flour", "all-purpose flour", "bread flour", "cake flour", "whole wheat flour",
    "cornstarch", "starch", "baking powder", "baking soda", "yeast",
    "vanilla", "vanilla extract", "almond extract", "cocoa", "cocoa powder",
    "cornmeal", "rolled oats", "oats", "breadcrumbs", "panko",
    "rice", "white rice", "brown rice", "jasmine rice", "basmati rice",
    "pasta", "noodles", "spaghetti", "macaroni", "tortilla", "wrap",
    "oil", "olive oil", "vegetable oil", "canola oil", "coconut oil",
    "butter", "margarine", "shortening", "lard",
    "honey", "maple syrup", "molasses", "corn syrup",
    "vinegar", "white vinegar", "apple cider vinegar", "balsamic vinegar",
    "soy sauce", "worcestershire sauce", "hot sauce", "mustard", "ketchup",
    "mayonnaise", "salsa", "broth", "stock", "bouillon",

    # Dairy
    "milk", "whole milk", "skim milk", "buttermilk", "cream", "heavy cream",
    "whipping cream", "sour cream", "yogurt", "greek yogurt", "buttermilk",
    "cheese", "cheddar", "parmesan", "mozzarella", "cream cheese", "cottage cheese",
    "butter", "eggs", "egg", "egg yolk", "egg white",

    # Produce
    "garlic", "onion", "red onion", "yellow onion", "white onion", "shallot",
    "leek", "scallion", "green onion", "chive",
    "carrot", "celery", "potato", "sweet potato", "yam", "turnip", "parsnip",
    "tomato", "tomatoes", "cucumber", "bell pepper", "pepper", "jalapeno",
    "mushroom", "spinach", "kale", "lettuce", "cabbage", "broccoli", "cauliflower",
    "zucchini", "squash", "pumpkin", "asparagus", "green bean", "pea", "peas",
    "corn", "avocado", "lemon", "lime", "orange", "grapefruit", "apple", "banana",
    "berry", "strawberry", "blueberry", "raspberry", "blackberry", "cherry",
    "grape", "pineapple", "mango", "peach", "pear", "plum", "apricot",
    "celery", "parsley", "cilantro", "coriander", "dill", "basil", "mint", "thyme",
    "rosemary", "oregano", "sage", "tarragon", "bay leaf",

    # Herbs and spices
    "ginger", "ground ginger", "cinnamon", "nutmeg", "clove", "allspice",
    "cardamom", "cumin", "paprika", "smoked paprika", "chili powder",
    "cayenne", "turmeric", "mustard powder", "garlic powder", "onion powder",
    "red pepper flakes", "crushed red pepper", "oregano", "basil", "thyme",
    "rosemary", "parsley", "coriander", "fennel", "anise", "sesame seeds",
    "poppy seeds", "sesame oil",

    # Proteins
    "chicken", "chicken breast", "chicken thighs", "ground chicken",
    "beef", "ground beef", "steak", "roast beef", "pork", "ground pork",
    "bacon", "ham", "sausage", "turkey", "ground turkey",
    "fish", "salmon", "tuna", "cod", "shrimp", "crab", "lobster",
    "tofu", "tempeh", "seitan", "beans", "black beans", "kidney beans",
    "chickpeas", "lentils",

    # Nuts, seeds, and add-ins
    "almonds", "walnuts", "pecans", "cashews", "peanuts", "pistachios",
    "hazelnuts", "macadamia nuts", "sunflower seeds", "pumpkin seeds",
    "chia seeds", "flaxseed", "sesame seeds", "raisins", "currants",
    "coconut", "shredded coconut", "chocolate chips", "chocolate",
    "dark chocolate", "white chocolate", "semisweet chocolate",

    # Baking-specific
    "corn syrup", "gelatin", "powdered gelatin", "molasses", "cream of tartar",
    "yeast", "active dry yeast", "instant yeast", "self-rising flour",
    "cake mix", "vanilla pudding", "food coloring", "sprinkles",

    # Spirits & Cocktail ingredients
    "tequila", "vodka", "gin", "whiskey", "bourbon", "scotch",
    "triple sec", "cointreau", "vermouth", "prosecco", "champagne",
    "coffee liqueur", "amaretto", "bitters", "espresso",
    "lemon juice", "lime juice", "orange juice",

    # Misc cooking ingredients
    "broth", "stock", "gravy", "salad dressing", "relish", "pickle", "pickles",
    "capers", "olives", "artichoke", "sun-dried tomatoes", "coconut milk",
    "cream of mushroom soup", "cream of chicken soup",
    "beer", "wine", "rum", "brandy"
]
COMMON_INGREDIENTS = list(set(COMMON_INGREDIENTS)) # deduplicate

MEASUREMENT_UNITS = [
    r"\bcups?\b", r"\bc\.\b",
    r"\btablespoons?\b", r"\btbsp\.?\b", r"(?<=\d)\s?T\b", 
    r"\bteaspoons?\b",  r"\btsp\.?\b",
    r"\bounces?\b",     r"\boz\.?\b",
    r"\bpounds?\b",     r"\blbs?\.?\b",
    r"\bgrams?\b",      r"(?<=\d)\s?g\b",
    r"\bkilograms?\b",  r"\bkg\b",
    r"\bmilliliters?\b",r"\bml\b",
    r"\bliters?\b",     r"\bl\b",
    r"\bpinch\b",       r"\bdash\b",    r"\bsmidgen\b",
    r"\bslices?\b",     r"\bcloves?\b",
    r"\bcans?\b",       r"\bpackages?\b", r"\benvelopes?\b",
    r"\bsticks?\b",     r"\bsprigs?\b",
    r"\bquarts?\b",     r"\bqt\.?\b",   r"\bpints?\b",   r"\bpt\.?\b",
    r"\bcontainer\b",   r"\bcarton\b",  r"\bbottle\b"
]

_KW_PATTERN = re.compile(
    r'\b(' + '|'.join(re.escape(k) for k in RECIPE_KEYWORDS) + r')\b',
    re.IGNORECASE
)


_UNIT_PATTERN = re.compile(
    '(' + '|'.join(MEASUREMENT_UNITS) + ')',
    re.IGNORECASE | re.VERBOSE
)

_FRACTION_PATTERN = re.compile(
    r'\b\d+\s*/\s*\d+\b|\b\d+\.\d+\b|[½⅓⅔¼¾⅕⅖⅗⅘⅙⅚⅛⅜⅝⅞]',
    re.UNICODE
)

_QTY_WORD_PATTERN = re.compile(r'\b\d+\s+[a-zA-Z]')


def _normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[-–—]", " ", text)   # normalize hyphens/dashes → space
    text = re.sub(r"\s+", " ", text)     # collapse whitespace
    return text.strip()


def _ingredient_to_pattern(ingredient: str) -> str:
    parts = ingredient.strip().split()
    
    # flexible spacing
    base = r'\s+'.join(re.escape(p) for p in parts)
    
    # simple plural
    plural = base + r's?'
    
    # lightweight irregulars
    irregular_map = {
        "tomato": "tomatoes?",
        "potato": "potatoes?",
        "leaf": "leaves?",
        "loaf": "loaves?",
    }
    
    last = parts[-1]
    
    if last in irregular_map:
        irregular = r'\s+'.join(
            [re.escape(p) for p in parts[:-1]] + [irregular_map[last]]
        )
        return f"(?:{plural}|{irregular})"
    
    return plural


_INGREDIENT_PATTERN = re.compile(
    r'\b(' + '|'.join(_ingredient_to_pattern(i) for i in COMMON_INGREDIENTS) + r')\b',
    re.IGNORECASE
)


def _clamp(value, lo=0.0, hi=1.0):
    return max(lo, min(hi, value))


def score_text(text: str) -> dict:
    if not text or not text.strip():
        return {"score": 0.0, "is_recipe": False, "signals": {}}

    text = _normalize_text(text)
    word_count = len(text.split())

    # 1. Ingredient Detection. # Each unique ingredient adds 0.04 to the score, capped at 0.30
    # We count unique ingredient mentions to prevent things like "sugar sugar sugar" being overly influential.
    ing_hits = len(set(_INGREDIENT_PATTERN.findall(text)))
    ing_score = _clamp(ing_hits * 0.05, 0, 0.40)

    # 2. Recipe keyword hits. Each keyword adds 0.05 to the score, capped at 0.35
    kw_hits = len(_KW_PATTERN.findall(text))
    kw_score = _clamp(kw_hits * 0.04, 0, 0.30)

    # 3. Measurement unit density
    unit_hits = len(_UNIT_PATTERN.findall(text))
    unit_density = unit_hits / max(word_count, 1)
    unit_score = _clamp(unit_density * 4.0, 0, 0.20)

    # 4. Fractions / quantities. Each fraction or quantity phrase adds 0.03 to the score, capped at 0.15
    frac_hits = len(_FRACTION_PATTERN.findall(text))
    qty_hits  = len(_QTY_WORD_PATTERN.findall(text))
    frac_score = _clamp((frac_hits + qty_hits) * 0.02, 0, 0.12)

    total = ing_score + kw_score + unit_score + frac_score

    return {
        "score": round(_clamp(total), 4),
        "is_recipe": total >= RECIPE_THRESHOLD,
        "signals": {
            "ingredient_score": round(ing_score, 3),
            "keyword_score": round(kw_score, 3),
            "unit_score": round(unit_score, 3),
            "fraction_score": round(frac_score, 3),
        }
    }


def extract_title(text: str) -> str | None:
    best_line = None
    highest_score = -1
    
    for line in text.splitlines()[:12]:
        line = line.strip()
        if len(line) < 4 or len(line) > 50:
            continue
            
        letters = [c for c in line if c.isalpha()]
        if not letters:
            continue
            
        caps_ratio = sum(1 for c in letters if c.isupper()) / len(letters)
        current_score = caps_ratio

        if current_score > highest_score:
            highest_score = current_score
            best_line = line
            
    return best_line


if __name__ == "__main__":
    samples = [
        """
Favorite Double Chocolate Chip Cookies Recipe

1/2 cup (8 Tbsp; 113g) unsalted butter, softened to room temperature
1/2 cup (100g) granulated sugar
1/2 cup (100g) packed light or dark brown sugar
1 large egg, at room temperature
1 teaspoon pure vanilla extract
1 cup (125g) all-purpose flour (spooned & leveled)
2/3 cup (55g) natural unsweetened cocoa powder
1 teaspoon baking soda
1/8 teaspoon salt
1 Tablespoon (15ml) milk (any kind, dairy or non)
1 and 1/4 cups (225g) semi-sweet chocolate chips, plus a few more for optional topping*

Instructions:
Preliminary note: This cookie dough requires at least 3 hours of chilling, but I prefer to chill the dough overnight. The colder the dough, the thicker the cookies.
In a large bowl using a hand-held or stand mixer fitted with a paddle attachment, beat the butter, granulated sugar, and brown sugar together on medium high speed until fluffy and light in color, about 3 minutes. (Here’s a helpful tutorial if you need guidance on how to cream butter and sugar.) Add the egg and vanilla extract, and then beat on high speed until combined. Scrape down the sides and bottom of the bowl as needed.
In a separate bowl, whisk the flour, cocoa powder, baking soda and salt together until combined. With the mixer running on low speed, slowly pour into the wet ingredients. Beat on low until combined. The cookie dough will be quite thick. Switch to high speed and beat in the milk, then the chocolate chips. The cookie dough will be sticky and tacky. Cover dough tightly and chill in the refrigerator for at least 3 hours and up to 3 days. Chilling is mandatory for this sticky cookie dough.
Remove cookie dough from the refrigerator and allow to sit at room temperature for 10 minutes. If the cookie dough chilled longer than 3 hours, let it sit at room temperature for about 20 minutes. This makes the chilled cookie dough easier to scoop and roll.
Preheat oven to 350°F (177°C). Line large baking sheets with parchment paper or silicone baking mats. (Always recommended for cookies.) Set aside.
Scoop and roll dough, a heaping 1.5 Tablespoons (about 35-40g; I like to use this medium cookie scoop) in size, into balls. To ensure a thicker cookie, make the balls taller than they are wide (almost like a cylinder or column). Arrange 2-3 inches apart on the baking sheets. The cookie dough is certainly sticky, so wipe your hands clean after every few balls of dough you shape.
Bake the cookies for 11-12 minutes or until the edges appear set and the centers still look soft. Tip: If they aren’t really spreading by minute 9, remove them from the oven and lightly bang the baking sheet on the counter 2-3x. This helps initiate that spread. Return to the oven to continue baking.
Cool cookies for 5 minutes on the baking sheet. During this time, I like to press a few more chocolate chips into the tops of the warm cookies. (This is optional and only for looks.) Transfer to cooling rack to cool completely. The cookies will slightly deflate as they cool. 
Cover leftover cookies tightly and store at room temperature for up to 1 week.
"""
    ]

    for sample in samples:
        result = score_text(sample)
        print(f"Score:     {result['score']}")
        print(f"Is recipe: {result['is_recipe']}")
        print(f"Signals:   {result['signals']}")
        print(f"Title:     {extract_title(sample)}")