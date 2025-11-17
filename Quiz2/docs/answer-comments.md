# QUIZ 2 Answers
## Charlie Gorman

### Q1 Comparing Energy Consumption
- a. AI Translator vs. Basic Translator (Google Translate): 
- The AI Translator (Large language model processing) uses 2.9Wh per query while Google Translate uses 0.3Wh per query.
- By default, Model size = Medium (LLaMA 7B), Batch Size = 1 task and Cloud Distance = Regional (same country).
- Next setting: Model Size = Tiny (DistilBERT), Batch Size = 1 task and Cloud Distance = Local (on device). This setting saves about 64% energy compared to typical cloud-based AI, 1Wh per query vs. 2.9Wh per query. 
- Next setting: Model Size = Huge (GPT-4), Batch Size = 10 tasks (batched) and Cloud Distance = Overseas (cross-continent). This setting saves about 71% energy compared to typical cloud-based AI, 0.8Wh per query vs. 2.9Wh per query. 
- b. Average Energy Difference:
- The average energy difference across the three settings checked was 1.267Wh. LLM is higher on average by 1.267Wh as well, across the three settings. 

### Content Cleaning
- I added titles to each section (ingredients, instructions, macronutrients, etc.) and spaced it so is easily parsible.  

### Running Code
- Stuffed Pepper (original recipe 1):

- PF1–PF3:

- python code/convert_pf.py --recipe-file data/original_recipe1.txt --recipe-name "One-Pan Stuffed Pepper Casserole" --url "https://www.allrecipes.com/one-pan-stuffed-pepper-casserole-recipe-11806363" --approach PF1 --out data/r3_outputs/pepper_PF1.json
- python code/convert_pf.py --recipe-file data/original_recipe1.txt --recipe-name "One-Pan Stuffed Pepper Casserole" --url "https://www.allrecipes.com/one-pan-stuffed-pepper-casserole-recipe-11806363" --approach PF2 --out data/r3_outputs/pepper_PF2.json
- python code/convert_pf.py --recipe-file data/original_recipe1.txt --recipe-name "One-Pan Stuffed Pepper Casserole" --url "https://www.allrecipes.com/one-pan-stuffed-pepper-casserole-recipe-11806363" --approach PF3 --out data/r3_outputs/pepper_PF3.json

- PP (stitched)

- python code/convert_pp.py --recipe-file data/original_recipe1.txt --recipe-name "One-Pan Stuffed Pepper Casserole" --url "https://www.allrecipes.com/one-pan-stuffed-pepper-casserole-recipe-11806363" --out data/r3_outputs/pepper_PP.json

- Turkey-Zucchini (original recipe 2):

- PF1–PF3:

- python code/convert_pf.py --recipe-file data/original_recipe2.txt --recipe-name "Ground Turkey and Zucchini Skillet" --url "https://www.allrecipes.com/ground-turkey-and-zucchini-skillet-recipe-11805473" --approach PF1 --out data/r3_outputs/turkey_PF1.json
- python code/convert_pf.py --recipe-file data/original_recipe2.txt --recipe-name "Ground Turkey and Zucchini Skillet" --url "https://www.allrecipes.com/ground-turkey-and-zucchini-skillet-recipe-11805473" --approach PF2 --out data/r3_outputs/turkey_PF2.json
- python code/convert_pf.py --recipe-file data/original_recipe2.txt --recipe-name "Ground Turkey and Zucchini Skillet" --url "https://www.allrecipes.com/ground-turkey-and-zucchini-skillet-recipe-11805473" --approach PF3 --out data/r3_outputs/turkey_PF3.json

- PP (stitched)

- python code/convert_pp.py --recipe-file data/original_recipe2.txt --recipe-name "Ground Turkey and Zucchini Skillet" --url "https://www.allrecipes.com/ground-turkey-and-zucchini-skillet-recipe-11805473" --out data/r3_outputs/turkey_PP.json


- Scoring all JSONs
- python code/score_r3.py data/r3_outputs/*.json


### Last Questions
- Q1: DIDNT GET TO ANSWER THESE QUESTIONS, BUT CAN BE RAN!
- Q2: 

