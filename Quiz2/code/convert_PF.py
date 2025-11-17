
import argparse, json, sys, re
from pathlib import Path
from prompts import PF1, PF2, PF3
from llm_client_hf import call_llm

PF_MAP = {"PF1": PF1, "PF2": PF2, "PF3": PF3}

def extract_json(text: str) -> str:
    """
    Extract the first JSON object/array from the model text (handles accidental prose).
    """
    # greedy object or array
    m = re.search(r'(\{[\s\S]*\}|\[[\s\S]*\])', text)
    return m.group(1).strip() if m else text

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recipe-file", required=True)
    ap.add_argument("--recipe-name", required=True)
    ap.add_argument("--url", required=True)
    ap.add_argument("--approach", required=True, choices=list(PF_MAP.keys()))
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=None)
    args = ap.parse_args()

    text = Path(args.recipe_file).read_text(encoding="utf-8")
    prompt = PF_MAP[args.approach].format(recipe_name=args.recipe_name, url=args.url, recipe_text=text)

    raw = call_llm(prompt, model=args.model)

    # Try to parse JSON; if not, try extracting JSON-looking section
    try_payloads = [raw, extract_json(raw)]
    data = None
    for payload in try_payloads:
        try:
            data = json.loads(payload)
            break
        except Exception:
            pass

    if data is None:
        Path(args.out + ".raw.txt").write_text(raw, encoding="utf-8")
        print("Output not valid JSON. Raw response saved.", file=sys.stderr)
        sys.exit(1)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {args.out}")

if __name__ == "__main__":
    main()
