###############################################################################
#       PRETRAINING / FINE-TUNING A GPT-2 MODEL USING HUGGING FACE
###############################################################################
#
# OVERVIEW
# --------
# This script demonstrates how to FINE-TUNE a pre-trained GPT-2 language model
# on a custom text corpus (Tiny Shakespeare) using the Hugging Face
# `transformers` library.
#
# KEY DISTINCTION -- Pre-training vs Fine-tuning:
#
#   PRE-TRAINING:
#       - Training a model FROM SCRATCH on a very large, general corpus
#         (e.g., all of Wikipedia, books, web pages).
#       - The model learns general language patterns: grammar, facts, reasoning.
#       - Requires enormous compute (hundreds of GPUs for days/weeks).
#
#   FINE-TUNING:
#       - Taking an ALREADY pre-trained model and continuing to train it on
#         a SMALLER, DOMAIN-SPECIFIC dataset (e.g., Shakespeare text).
#       - The model adapts its existing knowledge to the new style/domain.
#       - This is what we do in THIS script -- we load GPT-2's pre-trained
#         weights and fine-tune them on Shakespeare text.
#       - Much cheaper: can run on a single GPU or even CPU.
#
# WHY HUGGING FACE?
#   - Provides ready-to-use pre-trained models (GPT-2, BERT, T5, etc.)
#   - Handles tokenization, model architecture, weight loading, and generation
#     with simple, consistent APIs.
#   - The `transformers` library is the industry standard for NLP/LLM work.
#
###############################################################################


# =============================================================================
# MODULE 1: IMPORTS
# =============================================================================
#
# THEORY: Every deep learning pipeline needs these core building blocks:
#
# 1. torch (PyTorch)
#    - PyTorch is an open-source deep learning framework developed by Meta.
#    - It provides:
#      a) Tensors  -- multi-dimensional arrays (like NumPy but with GPU support)
#      b) Autograd -- automatic differentiation for computing gradients
#      c) nn       -- neural network layers (Linear, Embedding, Transformer, etc.)
#    - PyTorch uses a "define-by-run" (eager) execution model, meaning the
#      computation graph is built dynamically as code executes, making it
#      very Pythonic and easy to debug.
#
# 2. DataLoader & Dataset (from torch.utils.data)
#    - Dataset : an abstract class representing a dataset. You override:
#        __len__()       -> returns the number of samples
#        __getitem__(i)  -> returns the i-th sample
#    - DataLoader : wraps a Dataset and provides:
#        a) Automatic batching    -- groups samples into batches
#        b) Shuffling             -- randomizes order each epoch
#        c) Multi-process loading -- loads data in parallel (num_workers)
#        d) Collation             -- combines individual samples into a batch tensor
#
# 3. GPT2LMHeadModel (from transformers)
#    - This is the full GPT-2 model with a LANGUAGE MODELING HEAD on top.
#    - Architecture (simplified):
#        Input IDs -> Token Embeddings + Position Embeddings
#                  -> 12 Transformer Decoder Blocks (each with:
#                       - Masked Multi-Head Self-Attention
#                       - Feed-Forward Network
#                       - Layer Normalization
#                       - Residual Connections)
#                  -> Language Model Head (Linear layer mapping hidden states
#                                         to vocabulary logits)
#    - The "LMHead" part means it predicts the NEXT TOKEN at each position.
#    - GPT-2 "small" has ~124 Million parameters.
#
# 4. GPT2Tokenizer (from transformers)
#    - Converts raw text into token IDs that the model can process.
#    - GPT-2 uses Byte-Pair Encoding (BPE):
#        a) Starts with individual bytes/characters
#        b) Iteratively merges the most frequent pair of adjacent tokens
#        c) Builds a vocabulary of ~50,257 sub-word tokens
#    - Example: "unhappiness" -> ["un", "h", "app", "iness"] (approximate)
#    - BPE handles ANY text (even unseen words) by breaking it into known pieces.
#
# 5. math
#    - Standard Python library; used here to compute e^x for perplexity.
#
# =============================================================================

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import math


# =============================================================================
# MODULE 2: DEVICE SETUP
# =============================================================================
#
# THEORY: Hardware Acceleration for Deep Learning
#
# Deep learning involves massive matrix multiplications. Modern GPUs have
# thousands of cores optimized for parallel arithmetic, making them 10-100x
# faster than CPUs for training neural networks.
#
# torch.device("cuda")  -> NVIDIA GPU (uses CUDA toolkit)
# torch.device("cpu")   -> standard CPU
# torch.device("mps")   -> Apple Silicon GPU (M1/M2/M3 chips)
#
# HOW IT WORKS:
#   - torch.cuda.is_available() checks if an NVIDIA GPU + CUDA drivers exist
#   - If True  -> we train on GPU (much faster)
#   - If False -> we fall back to CPU (slower but always available)
#
# WHY THIS MATTERS:
#   - A model with 124M parameters needs to update all 124M values each step.
#   - On CPU this might take ~10 seconds per batch; on GPU, ~0.1 seconds.
#   - We must also move our data (tensors) to the SAME device as the model.
#     Mixing devices (e.g., model on GPU, data on CPU) causes runtime errors.
#
# =============================================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on: {device}")


# =============================================================================
# MODULE 3: TOKENIZER SETUP
# =============================================================================
#
# THEORY: What is Tokenization?
#
# Neural networks cannot process raw text directly -- they only understand
# numbers. Tokenization is the process of converting text into a sequence
# of integer IDs from a fixed vocabulary.
#
# TYPES OF TOKENIZATION:
#   1. Word-level:     "I love NLP"    -> [45, 892, 7623]
#      - Simple but huge vocabulary, can't handle unseen words (OOV problem)
#
#   2. Character-level: "I love NLP"   -> [73, 32, 108, 111, 118, 101, ...]
#      - Tiny vocabulary but very long sequences, loses word-level meaning
#
#   3. Sub-word (BPE):  "I love NLP"   -> [40, 1842, 32168]
#      - GPT-2 uses THIS approach (Byte-Pair Encoding)
#      - Balances vocabulary size (~50K) with sequence length
#      - Can represent ANY text by breaking unknown words into known sub-pieces
#
# GPT2Tokenizer.from_pretrained("gpt2"):
#   - Downloads the exact same tokenizer used during GPT-2's original training.
#   - Loads two files: vocab.json (token -> ID mapping) and merges.txt (BPE rules).
#   - Ensures our text is encoded identically to how GPT-2 was originally trained.
#
# PADDING TOKEN:
#   - When batching multiple sequences, they must all be the same length.
#   - Shorter sequences are "padded" with a special token to fill the gap.
#   - GPT-2 was NOT trained with a padding token (it's an autoregressive model
#     that processes one continuous stream of text).
#   - We assign eos_token (End-Of-Sequence, ID=50256) as the pad_token.
#   - This is a common workaround; the attention_mask tells the model to IGNORE
#     these padded positions so they don't affect the output.
#
# =============================================================================

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token


# =============================================================================
# MODULE 4: LOADING THE PRE-TRAINED GPT-2 MODEL
# =============================================================================
#
# THEORY: Transfer Learning & Pre-trained Models
#
# TRANSFER LEARNING is one of the most powerful ideas in modern AI:
#   1. Train a large model on a massive general dataset (pre-training)
#   2. Take that model and adapt it to your specific task (fine-tuning)
#
# This works because the model has already learned:
#   - English grammar and syntax
#   - Word meanings and relationships
#   - World facts and common sense
#   - How to generate coherent, fluent text
#
# GPT2LMHeadModel.from_pretrained("gpt2"):
#   - Downloads the PRE-TRAINED WEIGHTS from Hugging Face's model hub.
#   - These weights were trained on WebText (40GB of internet text) by OpenAI.
#   - The model has 12 transformer layers, 12 attention heads, 768-dim hidden size.
#   - Total: ~124 Million parameters.
#
# .to(device):
#   - Moves ALL model parameters (weights and biases) to the chosen device
#     (GPU or CPU).190 Every tensor in the model is transferred.
#
# WHY NOT TRAIN FROM SCRATCH?
#   - Training GPT-2 from scratch would require:
#       * ~40 GB of training data
#       * Multiple high-end GPUs
#       * Days to weeks of training time
#       * Significant cost ($10,000+ in cloud compute)
#   - Fine-tuning reuses all that effort and adapts it in minutes/hours.
#
# model.num_parameters():
#   - Returns the total count of trainable parameters.
#   - 124M parameters = 124 million floating-point numbers that define the model.
#   - Each parameter is a single learnable weight (e.g., a connection strength
#     in a neural network layer).
#
# =============================================================================

model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
print(f"Model parameters: {model.num_parameters() / 1e6:.2f} M")


# =============================================================================
# MODULE 5: DATASET PREPARATION
# =============================================================================
#
# THEORY: Custom Datasets for Language Modeling
#
# For language model training, the input and the label are the SAME sequence!
# This is called "self-supervised learning" or "next token prediction":
#
#   Input:  [The, cat, sat,  on ]  (tokens at positions 0, 1, 2, 3)
#   Label:  [cat, sat, on, the]   (the NEXT token at each position)
#
# The model learns to predict: given all tokens so far, what comes next?
# Internally, GPT2LMHeadModel shifts the labels by one position automatically,
# so we pass the SAME token IDs as both input_ids and labels.
#
# THE PYTORCH DATASET CLASS:
#   - We create a custom Dataset subclass called ShakespeareDataset.
#   - It wraps the tokenized encodings and serves individual samples.
#
#   __init__(self, encodings):
#       - Stores the tokenizer output (which contains input_ids, attention_mask, etc.)
#
#   __len__(self):
#       - Returns total number of samples (chunks of text).
#       - The DataLoader uses this to know when an epoch is complete.
#
#   __getitem__(self, idx):
#       - Returns a single sample as a dictionary with three keys:
#
#         a) 'input_ids': The tokenized text as integer IDs.
#            Example: [464, 3797, 3332, 319, ...] for "The cat sat on ..."
#
#         b) 'attention_mask': A binary mask (1s and 0s).
#            - 1 = real token (the model should attend to it)
#            - 0 = padding token (the model should IGNORE it)
#            This prevents padding tokens from influencing the model's predictions.
#
#         c) 'labels': Same as input_ids!
#            The model internally shifts these to create the prediction targets.
#            Position i predicts the token at position i+1.
#
#       - torch.tensor() converts Python lists to PyTorch tensors (GPU-compatible).
#
# =============================================================================

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


# =============================================================================
# MODULE 6: TEXT LOADING & TOKENIZATION
# =============================================================================
#
# THEORY: Preparing Text Data for Transformer Models
#
# STEP 1 -- Reading the corpus:
#   - We load a raw text file (e.g., Shakespeare's complete works).
#   - The entire text is read into a single Python string.
#   - encoding="utf-8" ensures special characters are handled correctly.
#
# STEP 2 -- Tokenization with chunking:
#   The tokenizer converts the entire text into overlapping chunks of token IDs.
#
#   Key parameters explained:
#
#   truncation=True:
#       - If a sequence exceeds max_length, it gets cut off.
#       - Without this, very long texts would cause memory errors.
#
#   max_length=256:
#       - Each chunk will be exactly 256 tokens long.
#       - GPT-2 supports up to 1024 tokens, but 256 is more memory-friendly.
#       - Shorter chunks = more chunks = faster iteration but less context per sample.
#
#   stride=128:
#       - The sliding window overlap between consecutive chunks.
#       - With max_length=256 and stride=128, chunks overlap by 128 tokens:
#           Chunk 1: tokens[0:256]
#           Chunk 2: tokens[128:384]
#           Chunk 3: tokens[256:512]
#           ...
#       - This overlap ensures no context is lost at chunk boundaries.
#       - Without overlap, the model would never see token transitions that
#         fall exactly at a boundary.
#
#   return_overflowing_tokens=True:
#       - Instead of returning ONLY the first chunk, return ALL chunks.
#       - Turns one long text into many training samples automatically.
#
#   padding="max_length":
#       - Pads shorter chunks (like the last one) to exactly 256 tokens.
#       - All chunks must be the same size for efficient batching.
#       - The padding tokens are marked as 0 in the attention_mask.
#
# STEP 3 -- Wrapping in Dataset and DataLoader:
#
#   Dataset:
#       - Wraps the encodings so PyTorch can access individual samples.
#
#   DataLoader:
#       - batch_size=8: Groups 8 samples into one batch for parallel processing.
#         * Larger batches -> more stable gradients, faster GPU utilization
#         * Smaller batches -> less memory usage, more frequent weight updates
#         * 8 is a good balance for fine-tuning on consumer hardware.
#
#       - shuffle=True: Randomizes the order of samples each epoch.
#         * Prevents the model from memorizing the order of the training data.
#         * Different random orderings each epoch improve generalization.
#         * Essential for proper stochastic gradient descent.
#
# =============================================================================

file_path = "input.txt"
with open(file_path, "r", encoding="utf-8") as f:
    text = f.read()

# Tokenize: convert raw text into overlapping chunks of token IDs
encodings = tokenizer(text, truncation=True, max_length=256,
                      stride=128, return_overflowing_tokens=True, padding="max_length")

dataset = ShakespeareDataset(encodings)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)


# =============================================================================
# MODULE 7: OPTIMIZER SETUP
# =============================================================================
#
# THEORY: Optimizers -- How Neural Networks Learn
#
# An OPTIMIZER is the algorithm that updates the model's weights based on
# the computed gradients (from backpropagation) to minimize the loss function.
#
# GRADIENT DESCENT (basic idea):
#   1. Forward pass: compute predictions and loss
#   2. Backward pass: compute gradients (how much each weight contributed to the error)
#   3. Update: move each weight in the OPPOSITE direction of its gradient
#      weight_new = weight_old - learning_rate * gradient
#
# AdamW OPTIMIZER (used here):
#   Adam = Adaptive Moment Estimation. It's an advanced optimizer that
#   maintains TWO moving averages for each parameter:
#     - m (first moment)  : tracks the average gradient direction
#     - v (second moment) : tracks the average gradient magnitude (squared)
#
#   This gives Adam several advantages over basic gradient descent:
#     a) Adaptive learning rates -- each parameter gets its own effective LR
#     b) Momentum -- smooths out noisy gradients for more stable training
#     c) Handles sparse gradients well
#
#   The "W" in AdamW stands for "Weight Decay" (decoupled):
#     - Weight decay is a regularization technique that slightly shrinks weights
#       each step to prevent overfitting.
#     - AdamW implements this CORRECTLY (decoupled from the gradient update),
#       unlike vanilla Adam which couples them incorrectly.
#
# LEARNING RATE (lr=5e-5 = 0.00005):
#   - The learning rate controls HOW MUCH the weights change each step.
#   - For FINE-TUNING, we use a VERY SMALL learning rate (5e-5).
#   - Why so small?
#       * The pre-trained weights already encode valuable knowledge.
#       * A large learning rate would violently overwrite that knowledge,
#         causing "catastrophic forgetting" -- the model forgets everything
#         it learned during pre-training.
#       * A tiny learning rate makes gentle, careful adjustments, preserving
#         the general knowledge while adapting to Shakespeare's style.
#   - Compare: pre-training from scratch typically uses lr=5e-4 (10x larger).
#
# =============================================================================

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)


# =============================================================================
# MODULE 8: THE FINE-TUNING (TRAINING) LOOP
# =============================================================================
#
# THEORY: The Training Loop -- Heart of Deep Learning
#
# Every neural network training follows this loop:
#
#   for each EPOCH:           (one full pass through the entire dataset)
#       for each BATCH:       (a subset of samples processed together)
#           1. FORWARD PASS   (compute predictions & loss)
#           2. BACKWARD PASS  (compute gradients via backpropagation)
#           3. WEIGHT UPDATE  (optimizer adjusts parameters)
#
# EPOCH:
#   - One complete pass through ALL training data.
#   - After each epoch, the model has seen every sample once.
#   - Multiple epochs (here, 3) allow the model to refine its learning.
#   - Too few epochs -> underfitting (hasn't learned enough)
#   - Too many epochs -> overfitting (memorizes training data, poor generalization)
#
# model.train():
#   - Puts the model into TRAINING MODE. This affects:
#       a) Dropout layers: randomly zero out neurons (regularization, prevents
#          overfitting by forcing redundancy in learned representations)
#       b) Batch normalization: uses batch statistics (not running averages)
#   - Always call model.train() before training, model.eval() before inference.
#
# INSIDE THE BATCH LOOP:
#
#   1. .to(device):
#      - Moves batch tensors to the same device (CPU/GPU) as the model.
#      - GPU operations require all tensors to be on the same GPU.
#
#   2. optimizer.zero_grad():
#      - CLEARS old gradients from the previous batch.
#      - PyTorch ACCUMULATES gradients by default (adds new to old).
#      - If we don't zero them, gradients from batch 1 would contaminate batch 2.
#      - Must be called BEFORE the forward pass of each new batch.
#
#   3. FORWARD PASS - outputs = model(input_ids, attention_mask, labels):
#      - The input token IDs flow through the entire GPT-2 architecture:
#          Token Embedding -> Position Embedding -> 12 Transformer Blocks -> LM Head
#      - Because we passed `labels`, the model also computes the LOSS internally:
#          * The LM Head outputs logits (raw scores) for each vocabulary token
#            at each position.
#          * The loss function is CROSS-ENTROPY LOSS:
#              - Compares the predicted probability distribution over the vocabulary
#                with the actual next token.
#              - Lower loss = the model is predicting the next token more accurately.
#              - Formula: Loss = -sum(y_true * log(y_predicted))
#
#   4. BACKWARD PASS - loss.backward():
#      - This is BACKPROPAGATION -- the core algorithm of deep learning.
#      - It computes the GRADIENT of the loss with respect to EVERY parameter
#        in the model (all 124M of them).
#      - The gradient tells us: "If I increase this weight slightly, how much
#        does the loss increase or decrease?"
#      - Uses the CHAIN RULE of calculus to propagate gradients backward
#        through the network, layer by layer.
#      - After this call, every parameter's .grad attribute is populated.
#
#   5. WEIGHT UPDATE - optimizer.step():
#      - The AdamW optimizer reads each parameter's .grad and updates the weight:
#          weight = weight - learning_rate * (adjusted_gradient + weight_decay * weight)
#      - This is where the model actually LEARNS -- weights shift to reduce loss.
#
#   6. total_loss += loss.item():
#      - loss.item() extracts the scalar loss value from the tensor.
#      - We accumulate it to compute the average loss over the epoch.
#      - .item() is important: it returns a plain Python float, releasing
#        the tensor from the computation graph (saves memory).
#
# EPOCH-LEVEL METRICS:
#
#   Average Loss (avg_loss):
#       - Total loss divided by number of batches.
#       - Tracks overall training progress. Should DECREASE over epochs.
#
#   Perplexity (math.exp(avg_loss)):
#       - A more intuitive metric for language models.
#       - Perplexity = e^(average_loss)
#       - Interpretation: "On average, the model is as confused as if it had
#         to choose uniformly among X tokens."
#       - Perplexity of 50 means: the model is as uncertain as choosing
#         randomly among 50 equally likely tokens.
#       - Lower perplexity = better model. GPT-2 achieves ~20-30 on general text.
#       - A perfect model would have perplexity = 1 (always predicts correctly).
#
# =============================================================================

epochs = 3
model.train()

for epoch in range(epochs):
    total_loss = 0

    for batch_idx, batch in enumerate(dataloader):
        # Move tensors to the correct device (CPU or GPU)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        # Clear accumulated gradients from the previous batch
        optimizer.zero_grad()

        # Forward pass: compute predictions and cross-entropy loss
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        # Backward pass: compute gradients via backpropagation
        loss.backward()

        # Weight update: optimizer adjusts all 124M parameters
        optimizer.step()

        # Accumulate loss for epoch-level metrics
        total_loss += loss.item()

        # Print progress every 10 batches
        if batch_idx % 10 == 0:
            print(f"Epoch {epoch} | Batch {batch_idx} | Loss: {loss.item():.4f}")

    # Compute and display epoch-level metrics
    avg_loss = total_loss / len(dataloader)
    perplexity = math.exp(avg_loss)
    print(f"--- Epoch {epoch} Complete | Avg Loss: {avg_loss:.4f} | Perplexity: {perplexity:.4f} ---")


# =============================================================================
# MODULE 9: TESTING THE FINE-TUNED MODEL (INFERENCE / TEXT GENERATION)
# =============================================================================
#
# THEORY: Inference & Autoregressive Text Generation
#
# After training, we test whether the model has learned Shakespeare's style.
#
# model.eval():
#   - Switches to EVALUATION MODE:
#       a) Dropout is DISABLED  (use all neurons for best predictions)
#       b) BatchNorm uses running statistics (not batch statistics)
#   - This ensures deterministic, reproducible output.
#
# tokenizer.encode(prompt, return_tensors='pt'):
#   - Converts the text prompt into token IDs.
#   - return_tensors='pt' returns a PyTorch tensor (not a Python list).
#   - Example: "O Romeo, Romeo! wherefore" -> tensor([[46, 22282, 11, 22282, 0, 49289]])
#
# torch.no_grad():
#   - A context manager that DISABLES gradient computation.
#   - During inference, we don't need gradients (we're not training).
#   - Benefits:
#       a) Reduces memory usage (no gradient tensors stored)
#       b) Speeds up computation (no backward graph built)
#       c) Should ALWAYS be used during inference.
#
# model.generate() -- AUTOREGRESSIVE TEXT GENERATION:
#   GPT-2 generates text ONE TOKEN AT A TIME:
#
#   Step 1: Feed the prompt tokens to the model.
#   Step 2: The model outputs a probability distribution over ALL 50,257 tokens
#           for the NEXT position.
#   Step 3: Select the next token (using sampling strategy).
#   Step 4: Append that token to the sequence.
#   Step 5: Repeat from Step 1 until max_length is reached or EOS token is generated.
#
#   Key parameters:
#
#   max_length=50:
#       - Generate at most 50 tokens total (prompt + new tokens).
#       - Controls the length of the generated text.
#
#   num_return_sequences=1:
#       - Generate 1 completion. Can be >1 to get multiple diverse outputs.
#
#   do_sample=True:
#       - Enable STOCHASTIC (random) sampling from the probability distribution.
#       - If False, uses GREEDY decoding (always picks the most probable token),
#         which tends to produce repetitive, boring text.
#       - Sampling introduces randomness, making output more creative and diverse.
#
#   temperature=0.8:
#       - Controls the RANDOMNESS of sampling.
#       - Temperature modifies the logits before softmax:
#           adjusted_logits = logits / temperature
#       - temperature < 1.0 (e.g., 0.8): SHARPENS the distribution
#           -> High-probability tokens become even more likely
#           -> More focused, coherent, but less creative output
#       - temperature = 1.0: No modification (raw model probabilities)
#       - temperature > 1.0 (e.g., 1.5): FLATTENS the distribution
#           -> All tokens become more equally likely
#           -> More random, creative, but potentially incoherent output
#       - 0.8 is a good balance: coherent but not too repetitive.
#
#   pad_token_id=tokenizer.eos_token_id:
#       - Tells the generate function which token ID to use for padding.
#       - Prevents a warning since GPT-2 has no default pad token.
#
# tokenizer.decode(generated[0], skip_special_tokens=True):
#   - Converts the generated token IDs BACK into human-readable text.
#   - skip_special_tokens=True removes any special tokens (like EOS/PAD)
#     from the output string.
#   - The BPE tokenizer reverses the encoding: token IDs -> sub-words -> text.
#
# EXPECTED RESULT:
#   - The output should sound like Shakespeare -- archaic vocabulary, poetic
#     structure, dramatic tone -- because the model was fine-tuned on his works.
#   - But it should also be COHERENT and grammatically correct, because the
#     model retains its pre-trained knowledge of English.
#   - This is the power of fine-tuning: domain-specific style + general knowledge.
#
# =============================================================================

model.eval()
prompt = "O Romeo, Romeo! wherefore"
input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)

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


# =============================================================================
# MODULE 10: SUMMARY -- THE COMPLETE PIPELINE AT A GLANCE
# =============================================================================
#
#   STEP 1: Import libraries (PyTorch, Hugging Face Transformers)
#   STEP 2: Set device (GPU if available, else CPU)
#   STEP 3: Load pre-trained tokenizer (GPT-2 BPE, 50,257 tokens)
#   STEP 4: Load pre-trained model (GPT-2, 124M parameters)
#   STEP 5: Prepare dataset (read text, tokenize into overlapping chunks)
#   STEP 6: Create DataLoader (batching, shuffling)
#   STEP 7: Set up optimizer (AdamW with tiny learning rate for fine-tuning)
#   STEP 8: Training loop (forward pass -> loss -> backward pass -> weight update)
#   STEP 9: Evaluate (generate text and verify Shakespeare style)
#
# KEY TAKEAWAYS:
#   - Fine-tuning adapts a powerful pre-trained model to a specific domain.
#   - Small learning rate prevents catastrophic forgetting.
#   - The model retains general English knowledge while learning Shakespeare's style.
#   - This same approach works for ANY text domain: legal, medical, code, etc.
#   - Hugging Face makes this entire pipeline simple with just a few API calls.
#
# =============================================================================