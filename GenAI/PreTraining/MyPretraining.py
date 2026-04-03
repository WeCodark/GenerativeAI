###############################################################################
#                  TRUE PRETRAINING FROM SCRATCH
#                  Using GPT-2 Architecture + Your Own Corpus ("myCorpus")
###############################################################################
#
# WHAT IS PRETRAINING?
# --------------------
# Pretraining means teaching a model to understand language FROM ZERO.
# The model starts with RANDOM weights -- it knows NOTHING.
# No grammar, no words, no meaning -- just random numbers.
#
# We feed it a HUGE amount of text (your "myCorpus") and the model slowly
# learns patterns:
#   - First it learns common letter combinations ("th", "ing", "tion")
#   - Then it learns words and their relationships
#   - Then it learns grammar rules
#   - Then it learns facts and reasoning
#
# This is DIFFERENT from fine-tuning:
#
#   FINE-TUNING:
#       model = GPT2LMHeadModel.from_pretrained("gpt2")   <-- starts SMART
#       Uses someone else's pre-trained brain, just teaches it a new style.
#
#   PRETRAINING (this file):
#       model = GPT2LMHeadModel(config)                   <-- starts BLANK
#       Builds the brain from scratch. YOU teach it everything.
#
# WHY WOULD YOU PRETRAIN FROM SCRATCH?
#   1. You have a language that GPT-2 doesn't know (Tamil, Kannada, etc.)
#   2. You have a very specialized domain (medical, legal, chemistry)
#   3. You want full control over what the model learns
#   4. Research purposes -- to understand how language models work
#
# IMPORTANT: Real pretraining needs:
#   - A VERY large corpus (gigabytes of text)
#   - Powerful GPUs (multiple A100s or H100s)
#   - Days or weeks of training time
#   - This script is structured for real pretraining but can run on small
#     data for learning purposes too.
#
###############################################################################


# =============================================================================
# MODULE 1: IMPORTS
# =============================================================================
#
# We need a few extra things compared to fine-tuning:
#
# 1. GPT2Config (NEW!)
#    - This is the BLUEPRINT for our model.
#    - It says: how many layers? how many attention heads? how big is the
#      vocabulary? how long can the input be?
#    - In fine-tuning, the config comes bundled with the pre-trained weights.
#    - In pretraining, WE define the config ourselves.
#
# 2. GPT2TokenizerFast (NEW!)
#    - A faster version of GPT2Tokenizer.
#    - We will TRAIN our OWN tokenizer on our corpus so it learns the
#      vocabulary of OUR data, not someone else's.
#
# 3. tokenizers library (NEW!)
#    - Hugging Face's low-level tokenizer training library.
#    - Lets us build a BPE tokenizer from scratch on our own text.
#
# 4. os and glob
#    - For reading files from the "myCorpus" folder.
#
# =============================================================================

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2LMHeadModel, GPT2Config, GPT2TokenizerFast
from tokenizers import ByteLevelBPETokenizer
import math
import os
import glob


# =============================================================================
# MODULE 2: DEVICE SETUP
# =============================================================================
#
# Same as fine-tuning -- pick GPU if available, otherwise CPU.
# For real pretraining you NEED a GPU (or multiple GPUs).
# CPU pretraining on a large corpus would take months.
#
# =============================================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on: {device}")


# =============================================================================
# MODULE 3: PREPARE YOUR CORPUS
# =============================================================================
#
# YOUR CORPUS STRUCTURE:
#   Imagine you have a folder called "myCorpus/" with many text files:
#
#   myCorpus/
#       book1.txt        (500 MB of literature)
#       book2.txt        (300 MB of articles)
#       wikipedia.txt    (2 GB of Wikipedia text)
#       news.txt         (1 GB of news articles)
#       ...
#
# The more text you have, the smarter your model will be.
#
# ROUGH GUIDELINES FOR CORPUS SIZE:
#   - 1 MB    -> model will barely learn anything useful
#   - 100 MB  -> model learns basic patterns and some words
#   - 1 GB    -> model starts understanding grammar
#   - 10 GB   -> model becomes reasonably capable
#   - 100 GB+ -> model approaches GPT-2 level quality
#
# For this script, we collect all .txt files from the "myCorpus" folder.
#
# =============================================================================

corpus_dir = "myCorpus"  # <-- Put your text files in this folder

# Collect all text file paths from the corpus directory
corpus_files = glob.glob(os.path.join(corpus_dir, "*.txt"))
print(f"Found {len(corpus_files)} text files in '{corpus_dir}/'")

# Quick sanity check -- make sure there are files to train on
if len(corpus_files) == 0:
    raise FileNotFoundError(
        f"No .txt files found in '{corpus_dir}/'. "
        "Please create this folder and put your text files inside it."
    )


# =============================================================================
# MODULE 4: TRAIN YOUR OWN TOKENIZER (FROM SCRATCH!)
# =============================================================================
#
# WHY TRAIN A NEW TOKENIZER?
#
#   GPT-2's tokenizer was built on English web text. It learned sub-words
#   that are common in THAT data (like "the", "ing", "tion").
#
#   If YOUR corpus is different (say, medical papers or a different language),
#   GPT-2's tokenizer would be a BAD FIT:
#     - Common words in your corpus might get split into many tiny pieces
#     - Rare English words that never appear in your data waste vocabulary space
#
#   Training your OWN tokenizer means:
#     - The vocabulary is optimized for YOUR data
#     - Common words/phrases in your corpus get their own tokens
#     - More efficient encoding = shorter sequences = faster training
#
# HOW BPE (Byte-Pair Encoding) TRAINING WORKS:
#
#   Step 1: Start with individual characters/bytes as the vocabulary.
#           Vocabulary: [a, b, c, d, e, ..., z, space, ., !, ...]
#
#   Step 2: Count all pairs of adjacent tokens in the corpus.
#           Most common pair might be: ("t", "h") -- because "th" is everywhere
#
#   Step 3: Merge that pair into a new token: "th"
#           Vocabulary: [a, b, c, ..., z, space, ., th]
#
#   Step 4: Repeat Steps 2-3 until you reach your desired vocabulary size.
#           After many merges: [a, b, ..., th, the, ing, tion, "the ", ...]
#
#   The result: common words become single tokens, rare words get split
#   into known pieces. This handles ANY text, even words never seen before.
#
# KEY PARAMETERS:
#
#   vocab_size=30000:
#       - How many unique tokens the vocabulary should have.
#       - Bigger vocabulary = each word uses fewer tokens, but model has more
#         parameters (embedding layer grows).
#       - GPT-2 uses 50,257. We use 30,000 as a reasonable starting point.
#       - You can adjust this based on your corpus size and diversity.
#
#   min_frequency=2:
#       - Only create a token if it appears at least 2 times in the corpus.
#       - Filters out extremely rare patterns that would waste vocabulary space.
#
#   special_tokens:
#       - <s>     : Start of sequence -- marks the beginning of a text
#       - <pad>   : Padding token -- fills shorter sequences to match batch length
#       - </s>    : End of sequence -- marks the end of a text
#       - <unk>   : Unknown token -- for characters not in our vocabulary
#       - <mask>  : Mask token -- used in masked language modeling (like BERT)
#
# =============================================================================

print("Training custom BPE tokenizer on your corpus...")

# Create a fresh BPE tokenizer (no pre-trained vocabulary)
bpe_tokenizer = ByteLevelBPETokenizer()

# Train the tokenizer on all your corpus files
bpe_tokenizer.train(
    files=corpus_files,
    vocab_size=30000,
    min_frequency=2,
    special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"]
)

# Save the trained tokenizer to disk (creates vocab.json and merges.txt)
tokenizer_save_dir = "my_tokenizer"
os.makedirs(tokenizer_save_dir, exist_ok=True)
bpe_tokenizer.save_model(tokenizer_save_dir)
print(f"Tokenizer saved to '{tokenizer_save_dir}/'")

# Load it back as a Hugging Face tokenizer (for compatibility with the model)
tokenizer = GPT2TokenizerFast.from_pretrained(tokenizer_save_dir)

# Set special tokens so the tokenizer knows how to pad and end sequences
tokenizer.pad_token = "<pad>"
tokenizer.bos_token = "<s>"
tokenizer.eos_token = "</s>"

print(f"Tokenizer vocabulary size: {tokenizer.vocab_size}")


# =============================================================================
# MODULE 5: DEFINE THE MODEL CONFIGURATION (THE BLUEPRINT)
# =============================================================================
#
# THIS IS THE BIG DIFFERENCE FROM FINE-TUNING!
#
# In fine-tuning, we do:
#     model = GPT2LMHeadModel.from_pretrained("gpt2")
#     This downloads BOTH the architecture AND the learned weights.
#
# In pretraining, we do:
#     config = GPT2Config(...)      <-- define the architecture
#     model = GPT2LMHeadModel(config)  <-- create model with RANDOM weights
#
# The model starts knowing NOTHING. All 124M+ weights are random numbers.
#
# CONFIG PARAMETERS EXPLAINED:
#
#   vocab_size (= tokenizer.vocab_size):
#       - Must match YOUR tokenizer's vocabulary size (30,000 in our case).
#       - This determines the size of the embedding layer:
#         A matrix of shape [vocab_size x hidden_size] = [30,000 x 768]
#       - Each row in this matrix is the "meaning vector" for one token.
#       - At the start, these vectors are random. After training, similar
#         words will have similar vectors.
#
#   n_positions (= 512):
#       - The MAXIMUM sequence length the model can handle.
#       - "How many tokens can the model look at in one go?"
#       - GPT-2 uses 1024, but 512 saves memory and is fine for most tasks.
#       - This creates positional embeddings: the model learns that position 0
#         is the start, position 1 is the second word, and so on.
#       - Longer = more context but MUCH more memory (attention is O(n^2)).
#
#   n_embd (= 768):
#       - The HIDDEN SIZE -- the dimension of each token's representation.
#       - "How many numbers describe each token's meaning?"
#       - Think of it as the "richness" of the model's understanding.
#       - 768 means each token is represented by a vector of 768 numbers.
#       - Bigger = more expressive but more parameters and slower.
#       - GPT-2 Small=768, Medium=1024, Large=1280, XL=1600.
#
#   n_layer (= 12):
#       - Number of TRANSFORMER BLOCKS stacked on top of each other.
#       - Each layer refines the token representations further:
#           Layer 1:  learns basic patterns (common word pairs)
#           Layer 6:  learns grammar and syntax
#           Layer 12: learns meaning and context
#       - More layers = deeper understanding but more compute.
#       - GPT-2 Small=12, Medium=24, Large=36, XL=48.
#
#   n_head (= 12):
#       - Number of ATTENTION HEADS in each transformer block.
#       - Multi-head attention lets the model focus on DIFFERENT things
#         at the same time:
#           Head 1 might focus on: subject-verb agreement
#           Head 2 might focus on: adjective-noun relationships
#           Head 3 might focus on: long-range references ("he" -> "John")
#       - Each head works with (n_embd / n_head) = 768/12 = 64 dimensions.
#       - n_head must evenly divide n_embd.
#
#   n_inner (= 3072):
#       - Size of the FEED-FORWARD NETWORK inside each transformer block.
#       - After attention combines information, the feed-forward network
#         processes it further.
#       - Usually set to 4 * n_embd = 4 * 768 = 3072 (this is standard).
#       - This is where a lot of the "thinking" happens -- factual knowledge
#         is believed to be stored in these feed-forward layers.
#
#   bos_token_id, eos_token_id, pad_token_id:
#       - Tell the model which token IDs correspond to special tokens.
#       - bos = beginning of sequence (token ID 0 = "<s>")
#       - eos = end of sequence (token ID 2 = "</s>")
#       - pad = padding (token ID 1 = "<pad>")
#       - These must match the IDs from YOUR tokenizer.
#
# TOTAL PARAMETERS CALCULATION (approximate):
#
#   Embedding layer:    vocab_size * n_embd              = 30,000 * 768   = 23M
#   Position embedding: n_positions * n_embd             = 512 * 768      = 0.4M
#   Each transformer block:
#       Attention:      4 * n_embd^2                     = 4 * 768^2      = 2.4M
#       Feed-forward:   2 * n_embd * n_inner             = 2 * 768 * 3072 = 4.7M
#       Layer norms:    4 * n_embd                       = 4 * 768        = 3K
#       Per block total:                                                  ~ 7.1M
#   12 blocks:          12 * 7.1M                                         = 85M
#   LM Head:            n_embd * vocab_size              = 768 * 30,000   = 23M
#   ------------------------------------------------------------------
#   TOTAL:              approximately 131 Million parameters
#
# =============================================================================

config = GPT2Config(
    vocab_size=tokenizer.vocab_size,
    n_positions=512,
    n_embd=768,
    n_layer=12,
    n_head=12,
    n_inner=3072,
    bos_token_id=tokenizer.convert_tokens_to_ids("<s>"),
    eos_token_id=tokenizer.convert_tokens_to_ids("</s>"),
    pad_token_id=tokenizer.convert_tokens_to_ids("<pad>"),
)

# Create the model with RANDOM weights (this is what makes it "pretraining")
model = GPT2LMHeadModel(config).to(device)

total_params = model.num_parameters()
print(f"Model created with RANDOM weights")
print(f"Total parameters: {total_params / 1e6:.2f} M")
print(f"Architecture: {config.n_layer} layers, {config.n_head} heads, {config.n_embd} hidden size")


# =============================================================================
# MODULE 6: DATASET CLASS FOR PRETRAINING
# =============================================================================
#
# HOW PRETRAINING DATA WORKS:
#
#   The model learns by predicting the NEXT TOKEN.
#
#   Given:  "The cat sat on the"
#   Target: "cat sat on the mat"
#
#   At each position, the model tries to predict what comes next.
#   The LOSS tells it how wrong it was, and backpropagation fixes the weights.
#
#   Over billions of these predictions, the model learns:
#     - "The" is often followed by a noun or adjective
#     - "sat on" is often followed by "the" + location
#     - How sentences are structured
#     - Facts about the world (if the corpus contains them)
#
# OUR DATASET CLASS:
#
#   We read ALL text files, combine them, and split into fixed-size chunks.
#
#   Each chunk is one training sample. The model sees the chunk and tries
#   to predict each next token within it.
#
#   block_size (= 512):
#       - How many tokens per chunk.
#       - Must be <= n_positions from the config.
#       - Longer blocks = more context = better learning, but more memory.
#
# =============================================================================

class PretrainingDataset(Dataset):
    def __init__(self, corpus_files, tokenizer, block_size=512):
        """
        Reads all corpus files and creates fixed-size token chunks.

        Args:
            corpus_files: List of paths to .txt files in your corpus.
            tokenizer: Your trained tokenizer.
            block_size: Number of tokens per training sample.
        """
        self.examples = []

        print("Tokenizing corpus files...")

        # Step 1: Read and combine all text files
        all_text = ""
        for file_path in corpus_files:
            with open(file_path, "r", encoding="utf-8") as f:
                all_text += f.read() + "\n"

        print(f"Total corpus size: {len(all_text) / 1e6:.2f} MB of text")

        # Step 2: Tokenize the entire text into one long list of token IDs
        #   This might take a while for large corpora!
        tokens = tokenizer.encode(all_text)
        print(f"Total tokens: {len(tokens):,}")

        # Step 3: Split into non-overlapping chunks of block_size
        #   We use non-overlapping chunks for pretraining (unlike fine-tuning)
        #   because the corpus is big enough that we don't need overlap.
        #
        #   Example with block_size=5:
        #     tokens = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110]
        #     chunk 1: [10, 20, 30, 40, 50]
        #     chunk 2: [60, 70, 80, 90, 100]
        #     (we drop [110] because it's less than block_size)
        #
        for i in range(0, len(tokens) - block_size, block_size):
            chunk = tokens[i : i + block_size]
            self.examples.append(chunk)

        print(f"Created {len(self.examples):,} training samples of {block_size} tokens each")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        """
        Returns one training sample.

        input_ids: the token sequence
        attention_mask: all 1s (no padding, every token is real)
        labels: same as input_ids (the model shifts them internally)
        """
        token_ids = torch.tensor(self.examples[idx], dtype=torch.long)
        return {
            'input_ids': token_ids,
            'attention_mask': torch.ones_like(token_ids),   # All 1s = no padding
            'labels': token_ids                              # Same as input (self-supervised)
        }


# =============================================================================
# MODULE 7: CREATE DATASET AND DATALOADER
# =============================================================================
#
# DATALOADER SETTINGS FOR PRETRAINING:
#
#   batch_size=16:
#       - How many samples to process at once.
#       - Bigger batch = more stable training, better GPU usage.
#       - But needs more GPU memory.
#       - Start with 16 and reduce if you run out of memory (OOM error).
#       - Big pretraining runs use batch sizes of 256-2048 (across many GPUs).
#
#   shuffle=True:
#       - Randomizes the order of chunks each epoch.
#       - Prevents the model from memorizing the order of the text.
#
#   num_workers=4:
#       - Uses 4 CPU processes to load data in parallel.
#       - While the GPU is training on one batch, the CPU prepares the next.
#       - This prevents the GPU from sitting idle waiting for data.
#       - Set to 0 if you face issues on Windows or small systems.
#
#   pin_memory=True:
#       - Pins data in CPU memory for faster transfer to GPU.
#       - A small optimization that speeds up CPU-to-GPU data movement.
#       - Only useful when training on GPU.
#
# =============================================================================

# Create the dataset (this reads and tokenizes all files)
dataset = PretrainingDataset(
    corpus_files=corpus_files,
    tokenizer=tokenizer,
    block_size=512
)

# Create the DataLoader
dataloader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)


# =============================================================================
# MODULE 8: OPTIMIZER AND LEARNING RATE SCHEDULER
# =============================================================================
#
# PRETRAINING OPTIMIZER:
#
#   We still use AdamW, but with a LARGER learning rate than fine-tuning.
#   Why? Because the model starts from RANDOM weights. There's no existing
#   knowledge to protect. We can make big, bold updates at the start.
#
#   lr=5e-4 (= 0.0005):
#       - 10x larger than fine-tuning (which uses 5e-5).
#       - From random weights, the model needs to make big jumps to learn.
#
#   weight_decay=0.01:
#       - Regularization -- prevents weights from growing too large.
#       - Helps the model generalize instead of memorizing.
#
# LEARNING RATE SCHEDULER (warmup + decay):
#
#   In pretraining, we DON'T use a constant learning rate. Instead:
#
#   Phase 1 -- WARMUP (first ~2000 steps):
#       Learning rate gradually increases from 0 to 5e-4.
#       Why? At the start, weights are random and gradients are noisy.
#       Starting with a tiny LR prevents the random gradients from
#       pushing the model in a bad direction.
#
#   Phase 2 -- CONSTANT or DECAY (rest of training):
#       After warmup, the LR stays at peak (or slowly decreases).
#       The model is now stable and can handle the full learning rate.
#
#   Visual:
#       LR
#       |        _______________
#       |       /               \
#       |      /                 \  (optional decay)
#       |     /                   \
#       |    /                     \
#       |___/                       \___
#       +------------------------------> Steps
#        warmup       main training
#
#   We use a simple linear warmup scheduler here.
#
# =============================================================================

from transformers import get_linear_schedule_with_warmup

# Optimizer with larger learning rate (since we start from scratch)
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=5e-4,
    weight_decay=0.01
)

# Calculate total training steps
epochs = 10  # Pretraining needs MORE epochs than fine-tuning
total_steps = epochs * len(dataloader)
warmup_steps = 2000  # Gradually increase LR for the first 2000 steps

# Create the learning rate scheduler
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps
)

print(f"Training plan:")
print(f"  Epochs:       {epochs}")
print(f"  Batches/epoch: {len(dataloader):,}")
print(f"  Total steps:  {total_steps:,}")
print(f"  Warmup steps: {warmup_steps:,}")


# =============================================================================
# MODULE 9: THE PRETRAINING LOOP
# =============================================================================
#
# This is similar to the fine-tuning loop but with some extra features
# needed for long pretraining runs:
#
# 1. GRADIENT CLIPPING (max_norm=1.0):
#    - Sometimes gradients become VERY large ("gradient explosion").
#    - This causes the weights to jump wildly and training becomes unstable.
#    - Gradient clipping limits the gradient magnitude to a maximum value.
#    - If the total gradient norm exceeds 1.0, all gradients are scaled down.
#    - This keeps training STABLE, especially early when weights are random.
#
# 2. LEARNING RATE SCHEDULER STEP:
#    - After each batch, we call scheduler.step() to update the learning rate.
#    - During warmup: LR increases slowly from 0 to 5e-4.
#    - After warmup: LR gradually decreases toward 0.
#
# 3. PERIODIC SAVING (checkpoints):
#    - Pretraining takes a LONG time (hours, days, or weeks).
#    - If your computer crashes, you lose ALL progress without checkpoints.
#    - We save the model every 5000 steps so you can resume from the last save.
#    - Each checkpoint saves: model weights, optimizer state, and scheduler state.
#
# 4. LOGGING:
#    - We print the loss every 100 steps (not every 10, because pretraining
#      has thousands of steps per epoch).
#    - We also print the current learning rate to see the warmup/decay.
#
# WHAT THE LOSS LOOKS LIKE DURING PRETRAINING:
#
#   Step 0:      Loss ~ 10.0  (random guessing among 30,000 tokens)
#                              ln(30000) = 10.3 -- this is pure chance!
#   Step 1000:   Loss ~ 7.0   (learning common words)
#   Step 5000:   Loss ~ 5.0   (learning basic grammar)
#   Step 20000:  Loss ~ 4.0   (learning more complex patterns)
#   Step 100000: Loss ~ 3.0   (learning meaning and context)
#
#   The loss will NEVER reach 0 -- language is inherently unpredictable.
#   A loss of 3.0 is quite good (perplexity ~ 20).
#
# =============================================================================

print("\n" + "=" * 60)
print("STARTING PRETRAINING FROM SCRATCH")
print("=" * 60 + "\n")

model.train()
global_step = 0

for epoch in range(epochs):
    total_loss = 0

    for batch_idx, batch in enumerate(dataloader):
        # Move data to GPU/CPU
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        # Clear old gradients
        optimizer.zero_grad()

        # Forward pass: predict next tokens and compute loss
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        loss = outputs.loss

        # Backward pass: compute gradients
        loss.backward()

        # Gradient clipping: prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Update weights
        optimizer.step()

        # Update learning rate (warmup / decay)
        scheduler.step()

        # Track loss
        total_loss += loss.item()
        global_step += 1

        # Print progress every 100 steps
        if global_step % 100 == 0:
            current_lr = scheduler.get_last_lr()[0]
            print(
                f"Step {global_step:>6} | "
                f"Epoch {epoch} | "
                f"Batch {batch_idx:>5} | "
                f"Loss: {loss.item():.4f} | "
                f"LR: {current_lr:.6f}"
            )

        # Save checkpoint every 5000 steps
        if global_step % 5000 == 0:
            checkpoint_dir = f"checkpoint-step-{global_step}"
            model.save_pretrained(checkpoint_dir)
            tokenizer.save_pretrained(checkpoint_dir)
            print(f"  >> Checkpoint saved to '{checkpoint_dir}/'")

    # End of epoch summary
    avg_loss = total_loss / len(dataloader)
    perplexity = math.exp(avg_loss) if avg_loss < 20 else float('inf')
    print(f"\n--- Epoch {epoch} Complete ---")
    print(f"    Average Loss: {avg_loss:.4f}")
    print(f"    Perplexity:   {perplexity:.2f}")
    print()


# =============================================================================
# MODULE 10: SAVE THE FINAL PRETRAINED MODEL
# =============================================================================
#
# After all epochs are done, we save the fully trained model.
#
# model.save_pretrained(path):
#   - Saves the model WEIGHTS to a folder.
#   - Creates two files:
#       config.json      -- the architecture settings (layers, heads, etc.)
#       pytorch_model.bin -- the actual weight values (hundreds of MB)
#
# tokenizer.save_pretrained(path):
#   - Saves the tokenizer alongside the model.
#   - Creates: vocab.json, merges.txt, tokenizer_config.json
#
# WHY SAVE BOTH?
#   - The model and tokenizer are a PAIR. They must match.
#   - The model expects token IDs from THIS specific tokenizer.
#   - If you use a different tokenizer, the IDs won't match and the
#     output will be garbage.
#
# AFTER SAVING, YOU CAN:
#   1. Load your model later:
#       model = GPT2LMHeadModel.from_pretrained("my_pretrained_gpt2")
#       tokenizer = GPT2TokenizerFast.from_pretrained("my_pretrained_gpt2")
#
#   2. Fine-tune it on a specific task (just like people fine-tune GPT-2!)
#
#   3. Share it on Hugging Face Hub for others to use.
#
# =============================================================================

save_dir = "my_pretrained_gpt2"
model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)
print(f"\nFinal model saved to '{save_dir}/'")
print("You can now load this model with: GPT2LMHeadModel.from_pretrained('my_pretrained_gpt2')")


# =============================================================================
# MODULE 11: TEST YOUR PRETRAINED MODEL (GENERATE TEXT)
# =============================================================================
#
# Let's see what our model learned!
#
# WHAT TO EXPECT:
#
#   After very little training (small corpus, few epochs):
#     - Output will be mostly nonsense
#     - But you might see some real words appearing
#
#   After moderate training:
#     - Real words and partial grammar
#     - Some meaningful phrases mixed with gibberish
#
#   After extensive training (large corpus, many epochs):
#     - Coherent sentences
#     - Proper grammar
#     - Relevant content
#
# The quality depends entirely on:
#   1. How much data you trained on (more = better)
#   2. How long you trained (more epochs = better, up to a point)
#   3. The model size (bigger = more capable, but needs more data)
#
# =============================================================================

model.eval()

# Use a simple prompt to test
prompt = "The"
input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)

print("\n" + "=" * 60)
print("TEXT GENERATION TEST")
print("=" * 60)

with torch.no_grad():
    generated = model.generate(
        input_ids,
        max_length=100,
        num_return_sequences=3,         # Generate 3 different completions
        do_sample=True,
        temperature=0.9,
        top_k=50,                       # Only sample from top 50 tokens
        top_p=0.95,                     # Nucleus sampling (see explanation below)
        pad_token_id=tokenizer.convert_tokens_to_ids("<pad>"),
    )

# top_k=50:
#   - Instead of sampling from ALL 30,000 tokens, only consider the
#     TOP 50 most probable tokens.
#   - This removes very unlikely tokens (like random symbols) from the
#     sampling pool, making output more coherent.
#
# top_p=0.95 (nucleus sampling):
#   - Another way to filter tokens: keep the smallest set of tokens whose
#     combined probability is >= 95%.
#   - Example: if "the" has 40% probability, "a" has 30%, "this" has 20%,
#     and "an" has 10% -- top_p=0.95 would keep "the", "a", "this" (total=90%)
#     and add "an" to reach 100%. Other tokens with tiny probabilities are removed.
#   - More adaptive than top_k because it adjusts based on the distribution.

for i, output in enumerate(generated):
    text = tokenizer.decode(output, skip_special_tokens=True)
    print(f"\nGeneration {i + 1}:")
    print(f"  {text}")


# =============================================================================
# MODULE 12: SUMMARY -- PRETRAINING vs FINE-TUNING COMPARISON
# =============================================================================
#
# +------------------+---------------------------+---------------------------+
# |                  | PRETRAINING (this file)   | FINE-TUNING (other file)  |
# +------------------+---------------------------+---------------------------+
# | Starting weights | RANDOM (knows nothing)    | PRE-TRAINED (already smart)|
# | Tokenizer        | Train YOUR OWN            | Use existing GPT-2 one    |
# | Config           | YOU define architecture    | Comes with pre-trained    |
# | Learning rate    | Large (5e-4)              | Tiny (5e-5)               |
# | Epochs           | Many (10-100+)            | Few (2-5)                 |
# | Data needed      | Huge (GBs of text)        | Small (MBs of text)       |
# | Compute needed   | Multiple GPUs, days/weeks | Single GPU, minutes/hours |
# | Gradient clip    | Yes (training is unstable)| Usually not needed        |
# | LR scheduler     | Warmup + decay (important)| Optional                  |
# | Checkpoints      | Essential (long training)  | Nice to have              |
# | Goal             | Learn language from zero  | Adapt to a specific style |
# +------------------+---------------------------+---------------------------+
#
# THE PRETRAINING PIPELINE (what we did):
#
#   STEP 1:  Collect your corpus ("myCorpus/" folder with text files)
#   STEP 2:  Train a custom BPE tokenizer on your corpus
#   STEP 3:  Define model architecture with GPT2Config
#   STEP 4:  Create model with RANDOM weights
#   STEP 5:  Tokenize corpus into fixed-size chunks
#   STEP 6:  Set up optimizer (AdamW) + LR scheduler (warmup + decay)
#   STEP 7:  Train for many epochs with gradient clipping
#   STEP 8:  Save checkpoints periodically
#   STEP 9:  Save final model + tokenizer
#   STEP 10: Test with text generation
#
# AFTER PRETRAINING:
#   Your model is now a "base model" -- it understands YOUR corpus's language.
#   You can then FINE-TUNE it on specific tasks, just like people fine-tune GPT-2!
#
# =============================================================================
