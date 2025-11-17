
import argparse, json, re, sys
from pathlib import Path
from prompts import PP_ING, PP_INST
from llm_client_hf import call_llm

def extract_json(text: str) -> str:
    m = re.search(r'(\{[\s\S]*\}|\[[\s\S]*\])', text)
    return m.group(1).strip() if m else text

def parse_json_or_die(s: str, raw_path: str):
    try:
        return json.loads(s)
    except Exception:
        Path(raw_path).write_text(s, encoding="utf-8")
        print(f"Non-JSON from model. Saved raw to {raw_path}", file=sys.stderr)
        sys.exit(1)

def try_parse_macros(text: str):
    # Parse lines like "351 Calories", "22g Fat", "16g Carbs", "23g Protein"
    cal = re.search(r"(\d+)\s*Cal", text, re.I)
    fat = re.search(r"(\d+)\s*g\s*Fat", text, re.I)
    carbs = re.search(r"(\d+)\s*g\s*Carb", text, re.I)
    prot = re.search(r"(\d+)\s*g\s*Prot", text, re.I)
    if not any([cal, fat, carbs, prot]):
        return None
    val = lambda m: float(m.group(1)) if m else None
    return {"calories": val(cal), "fat_g": val(fat), "carbs_g": val(carbs), "protein_g": val(prot)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recipe-file", required=True)
    ap.add_argument("--recipe-name", required=True)
    ap.add_argument("--url", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=None)
    args = ap.parse_args()

    text = Path(args.recipe_file).read_text(encoding="utf-8")

    # INGREDIENTS
    ing_raw = call_llm(PP_ING.format(recipe_text=text), model=args.model)
    ing_json = extract_json(ing_raw)
    ingredients = parse_json_or_die(ing_json, args.out + ".ingredients.raw.txt")

    # INSTRUCTIONS
    inst_raw = call_llm(PP_INST.format(recipe_text=text), model=args.model)
    inst_json = extract_json(inst_raw)
    instructions = parse_json_or_die(inst_json, args.out + ".instructions.raw.txt")

    r3 = {
        "recipe_name": args.recipe_name,
        "data_provenance": args.url,       # per your testcase: a string URL
        "macronutrients": try_parse_macros(text),  # None if not found
        "ingredients": ingredients,
        "instructions": instructions
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(r3, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {args.out}")

if __name__ == "__main__":
    main()
