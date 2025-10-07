
import sys, json

REQ_KEYS = ["recipe_name","data_provenance","macronutrients","ingredients","instructions"]

def score_file(path: str) -> int:
    try:
        j = json.load(open(path, "r", encoding="utf-8"))
        score = 50
    except Exception:
        return 0
    for k in REQ_KEYS:
        if k in j:
            score += 10
    return min(score, 100)

if __name__ == "__main__":
    for p in sys.argv[1:]:
        print(f"{p}: {score_file(p)}")
