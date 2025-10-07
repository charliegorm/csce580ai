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
