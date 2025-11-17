PF1 = """You are converting a web recipe to an R3 JSON for a robotic chef.

Constraints:
- Output ONLY valid JSON, with keys: ["recipe_name","data_provenance","macronutrients","ingredients","instructions"].
- "data_provenance" MUST be the URL string (not an object).
- Each instruction is a short, single-action sentence.
- Keep original units/quantities; if unknown, use null (not "N/A").
- "macronutrients" can be null OR an object with numeric fields: {{ "calories":..., "fat_g":..., "carbs_g":..., "protein_g":... }}

INPUT:
Recipe Name: {recipe_name}
Data Provenance (URL): {url}

Text:
<<<
{recipe_text}
>>>"""

PF2 = """Task: Convert the recipe below to strict R3 JSON. Respond ONLY with JSON.

Schema (exact keys):
- recipe_name: string
- data_provenance: string (URL)
- macronutrients: null OR {{ "calories":number,"fat_g":number,"carbs_g":number,"protein_g":number }}
- ingredients: array of objects [{{"name":string, "quantity":number|null, "unit":string|null, "preparation":string|null}}]
- instructions: array of strings (each a single, simple action)

INPUT:
Recipe Name: {recipe_name}
Data Provenance (URL): {url}

Text:
<<<
{recipe_text}
>>>"""

PF3 = """Convert this recipe to R3 JSON by:
1) Extracting title, ingredients (name, qty, unit, preparation), and single-action instructions
2) Emitting ONLY JSON with keys:
   "recipe_name", "data_provenance", "macronutrients", "ingredients", "instructions"
3) Set data_provenance to the URL (string). macronutrients may be null or numeric object like: {{ "calories":..., "fat_g":..., "carbs_g":..., "protein_g":... }}

Recipe Name: {recipe_name}
Data Provenance (URL): {url}

TEXT:
<<<
{recipe_text}
>>>"""

PP_ING = """Extract ONLY ingredients as a JSON array of objects:
[
  {{ "name": "...", "quantity": <number or null>, "unit": "<string or null>", "preparation": "<string or null>" }}
]
No extra keys or comments.

TEXT:
<<<
{recipe_text}
>>>"""

PP_INST = """Extract ONLY instructions as a JSON array of strings.
Rules:
- Each element is a short, single-action sentence.
- Do not include step numbers in the text, just the sentences.

TEXT:
<<<
{recipe_text}
>>>"""
