import torch
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import math

# 1. Setup Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on: {device}")

# 2. Tokenizer Setup
# We load the pre-trained tokenizer that already knows GPT-2's vocabulary
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
# GPT-2 doesn't have a default padding token, so we assign the End-Of-Sequence token to it
tokenizer.pad_token = tokenizer.eos_token


# 3.Load the Pre-Trained Model

# Instead of initializing from a blank config, we download the actual weights!
# This model already understands English, grammar, and facts.
model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
print(f"Model parameters: {model.num_parameters() / 1e6:.2f} M")


# 4. Dataset Preparation (Tiny Shakespeare)
class ShakespeareDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return len(self.encodings.input_ids)

    def __getitem__(self, idx):
        return {
            'input_ids': torch.tensor(self.encodings.input_ids[idx]),
            'attention_mask': torch.tensor(self.encodings.attention_mask[idx]),
            'labels': torch.tensor(self.encodings.input_ids[idx])
        }

# Read the local text file (e.g., your Shakespeare text)
file_path = "input.txt"
with open(file_path, "r", encoding="utf-8") as f:
    text = f.read()

# Chunk the text into blocks of 256 tokens
encodings = tokenizer(text, truncation=True, max_length=256,
                      stride=128, return_overflowing_tokens=True, padding="max_length")

dataset = ShakespeareDataset(encodings)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True) # Smaller batch size for the larger pre-trained model


# 5. Optimizer
# Notice the learning rate is MUCH smaller (5e-5 instead of 5e-4)
# We don't want to drastically overwrite the model's existing knowledge!
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

# 6. The Fine-Tuning Loop
epochs = 3
model.train()

for epoch in range(epochs):
    total_loss = 0

    for batch_idx, batch in enumerate(dataloader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        # Backward pass
        loss.backward()

        # Update weights
        optimizer.step()

        total_loss += loss.item()

        if batch_idx % 10 == 0:
            print(f"Epoch {epoch} | Batch {batch_idx} | Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(dataloader)
    perplexity = math.exp(avg_loss)
    print(f"--- Epoch {epoch} Complete | Avg Loss: {avg_loss:.4f} | Perplexity: {perplexity:.4f} ---")

# 7. Test the Fine-Tuned Model
model.eval()
prompt = "O Romeo, Romeo! wherefore"
input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)

# We generate text. Notice it will sound like Shakespeare, but it leverages
# its pre-trained understanding of English to make the sentences much more coherent!
with torch.no_grad():
    generated = model.generate(
        input_ids,
        max_length=50,
        num_return_sequences=1,
        do_sample=True,
        temperature=0.8,
        pad_token_id=tokenizer.eos_token_id
    )

print("\n--- Fine-Tuned Generation Test ---")
print(tokenizer.decode(generated[0], skip_special_tokens=True))