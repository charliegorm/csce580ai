# using huggingface for llm
from transformers import pipeline

# Default: small-ish instruct model that runs on CPU reasonably
_DEFAULT_MODEL = "microsoft/Phi-3-mini-4k-instruct"

# Keep one global pipeline to avoid reloading the model each call
_PIPE = None

def get_pipe(model: str | None = None):
    global _PIPE
    if _PIPE is None or (model and getattr(_PIPE, "model_name", None) != model):
        chosen = model or _DEFAULT_MODEL
        _PIPE = pipeline(
            "text-generation",
            model=chosen,
            # trust_remote_code lets HF run custom generate() for chatty models when needed
            trust_remote_code=True
        )
        _PIPE.model_name = chosen
    return _PIPE

def call_llm(prompt: str, model: str | None = None, max_new_tokens: int = 800) -> str:
    """
    Returns text from HF pipeline. For best JSON results, prompts must request JSON explicitly.
    """
    pipe = get_pipe(model)
    out = pipe(
        prompt,
        max_new_tokens=max_new_tokens,
        do_sample=False,          # more deterministic JSON
        temperature=0.2,
        pad_token_id=pipe.tokenizer.eos_token_id
    )
    # Some pipelines return list[dict]; grab text
    text = out[0].get("generated_text", "")
    # Remove the leading prompt echo if present
    if text.startswith(prompt):
        text = text[len(prompt):]
    return text.strip()
