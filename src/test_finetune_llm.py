# src/test_finetune_llm.py

import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

device = "cuda" if torch.cuda.is_available() else "cpu"

# 1. Load Fine-Tuned Model
model_path = "/content/drive/MyDrive/pooosaaaaa/fine_tuned_azure_llm"

tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path)
model.to(device)
model.eval()

# 2. Generate text
@torch.no_grad()
def generate_text(prompt, max_new_tokens=100):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs.input_ids

    generated_ids = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=True,              # Sampling makes output more natural
        top_k=50,                    # Only choose among top 50 options
        top_p=0.95,                  # Cumulative probability sampling
        temperature=0.7,              # Lower temp = more focused
        pad_token_id=tokenizer.eos_token_id
    )

    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return generated_text

# 3. Main test
if __name__ == "__main__":
    prompt = "I have a Java application with SQL Server database. How can I migrate to Azure?"
    result = generate_text(prompt)

    print("\n=== Prompt ===")
    print(prompt)
    print("\n=== Generated Response ===")
    print(result)
