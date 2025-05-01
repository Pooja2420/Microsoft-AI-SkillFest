# src/train_finetune_llm.py

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import matplotlib.pyplot as plt
import os
import glob

device = "cuda" if torch.cuda.is_available() else "cpu"

# 1. Custom Dataset
class AzureTextDataset(Dataset):
    def __init__(self, data_folder, tokenizer, block_size=512):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.data = ""

        # Load all text
        text_files = glob.glob(os.path.join(data_folder, "*.txt"))
        for file in text_files:
            with open(file, "r", encoding="utf-8") as f:
                self.data += f.read() + "\n"

        # Tokenize
        encodings = self.tokenizer(self.data, return_tensors='pt', truncation=False)
        self.input_ids = encodings.input_ids[0]

    def __len__(self):
        return len(self.input_ids) // self.block_size

    def __getitem__(self, idx):
        start = idx * self.block_size
        end = start + self.block_size
        x = self.input_ids[start:end]
        y = self.input_ids[start+1:end+1]
        return x, y

# 2. Fine-Tune Function
def train(model, tokenizer, dataset, output_dir="./saved_models", epochs=3, batch_size=2, lr=5e-5):
    model.train()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = AdamW(model.parameters(), lr=lr)
    loss_history = []

    for epoch in range(epochs):
        total_loss = 0
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs, labels=targets)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        loss_history.append(avg_loss)

        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

        # Save checkpoint
        os.makedirs(output_dir, exist_ok=True)
        model.save_pretrained(f"{output_dir}/checkpoint_epoch_{epoch+1}")
        tokenizer.save_pretrained(f"{output_dir}/checkpoint_epoch_{epoch+1}")
        print(f"Checkpoint saved at {output_dir}/checkpoint_epoch_{epoch+1}")

    # Plot loss curve
    plt.plot(range(1, epochs+1), loss_history, marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Average Loss")
    plt.title("Fine-tuning Loss Curve")
    plt.grid(True)
    plt.savefig(f"{output_dir}/training_loss_curve.png")
    plt.show()

    print(f"Final model and tokenizer saved to {output_dir}")

# 3. Main
def main():
    data_folder = "/content/drive/MyDrive/pooosaaaaa/data_scraped_all"

    tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')
    model = GPT2LMHeadModel.from_pretrained('distilgpt2')
    model.resize_token_embeddings(len(tokenizer))  # important if you add new tokens
    model.to(device)

    dataset = AzureTextDataset(data_folder, tokenizer, block_size=512)

    train(
        model, tokenizer, dataset,
        output_dir="/content/drive/MyDrive/pooosaaaaa/fine_tuned_azure_llm",
        epochs=3,
        batch_size=2,
        lr=5e-5
    )

if __name__ == "__main__":
    main()
