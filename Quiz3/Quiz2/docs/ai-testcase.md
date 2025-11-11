# AI Test Case: Recipe → R3 JSON Conversion

## Objective
Convert two public food recipes from text into the R3 semi-structured JSON format.
---

## Inputs
Recipes Selected:
1. One-Pan Stuffed Pepper Casserole  
   Source: [Allrecipes](https://www.allrecipes.com/one-pan-stuffed-pepper-casserole-recipe-11806363)

2. Ground Turkey and Zucchini Skillet  
   Source: [Allrecipes](https://www.allrecipes.com/ground-turkey-and-zucchini-skillet-recipe-11805473)

Input Files:
- `./data/original_recipe1.txt` — cleaned text for Stuffed Pepper Casserole  
- `./data/original_recipe2.txt` — cleaned text for Ground Turkey and Zucchini Skillet  
---

## Methods
Both Prompt-Full (PF) and Prompt-Partial (PP) prompting strategies are used.

### Prompt-Full Approach (PF1, PF2, PF3)
Each prompt instructs the LLM (ChatGPT) to generate a complete R3 JSON in a single call.

- PF1: strict JSON schema instructions and single-action step enforcement  
- PF2: schema with built-in key validation  
- PF3: self-contained extract-and-assemble prompt  

---

### Prompt-Partial Approach (PP-1, PP-2)
LLM is prompted separately for:

- PP-1: ingredients list only  
- PP-2: instructions only  

The outputs are stitched together using a Python script (`./code/pp_stitch_r3.py`) to form the final combined PP JSON* for each recipe.  
---

## R3 Schema Fields (Minimum Required)
Each JSON output must include:

1. `recipe_name`  
2. `data_provenance`   
3. `macronutrients`   
4. `ingredients` (array of objects with `name`, `quantity`, `unit`, `preparation`)  
5. `instructions` (array of short, single-action sentences)

---

## Evaluation Rubric / Goodness Score
Implemented in `./code/score_r3.py`

---

## Evaluation Procedure
1. Run the LLM with each prompt variant  
2. Save outputs to `./data/r3_outputs/`  
3. Score each output:
   ```bash
   python code/score_r3.py data/r3_outputs/*.json
