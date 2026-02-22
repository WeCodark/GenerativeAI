import tensorflow as tf   # TensorFlow deep learning library
import numpy as np        # Numerical operations (arrays, matrices)
import math               # Mathematical functions

# Attention Paper: https://arxiv.org/pdf/1706.03762
# =========================================================
# 1. POSITIONAL ENCODING
# Transformer does not know word order.
# So we manually add position information using sine/cosine.
# =========================================================

class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, embed_dim, max_seq_len=500):
        super(PositionalEncoding, self).__init__()
        
        self.embed_dim = embed_dim
        
        # Create empty matrix to store positional encodings
        # Shape: (max_seq_len, embed_dim)
        pe = np.zeros((max_seq_len, embed_dim))
        
        # Position numbers: 0,1,2,3...
        position = np.arange(0, max_seq_len)[:, np.newaxis]
        
        # Scaling term used in original Transformer paper
        div_term = np.exp(
            np.arange(0, embed_dim, 2) * -(math.log(10000.0) / embed_dim)
        )
        
        # Apply sine to even index dimensions
        pe[:, 0::2] = np.sin(position * div_term)
        
        # Apply cosine to odd index dimensions
        pe[:, 1::2] = np.cos(position * div_term)
        
        # Add batch dimension and convert to TensorFlow tensor
        self.pe = tf.cast(pe[np.newaxis, :, :], dtype=tf.float32)

    def call(self, x):
        # x shape: (batch_size, seq_len, embed_dim)
        
        # Get sequence length dynamically
        seq_len = tf.shape(x)[1]
        
        # Add positional encoding to word embeddings
        return x + self.pe[:, :seq_len, :]


# =========================================================
# 2. MULTI HEAD ATTENTION
# Model learns which words should focus on which other words.
# =========================================================

class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        # Each head dimension size
        self.head_dim = embed_dim // num_heads
        
        # Ensure divisible
        assert self.head_dim * num_heads == embed_dim
        
        # Linear layers to generate Q, K, V
        self.q_linear = tf.keras.layers.Dense(embed_dim)
        self.k_linear = tf.keras.layers.Dense(embed_dim)
        self.v_linear = tf.keras.layers.Dense(embed_dim)
        
        # Final output projection
        self.out_linear = tf.keras.layers.Dense(embed_dim)

    def split_heads(self, x, batch_size):
        """
        Split embedding into multiple attention heads
        """
        
        # New shape: (batch, seq_len, num_heads, head_dim)
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.head_dim))
        
        # Move heads dimension forward
        return tf.transpose(x, perm=[0, 2, 1, 3])
        # Output: (batch, heads, seq_len, head_dim)

    def call(self, q, k, v, mask=None):
        
        batch_size = tf.shape(q)[0]

        # Step 1: Create Q, K, V matrices
        Q = self.split_heads(self.q_linear(q), batch_size)
        K = self.split_heads(self.k_linear(k), batch_size)
        V = self.split_heads(self.v_linear(v), batch_size)

        # Step 2: Attention score calculation
        matmul_qk = tf.matmul(Q, K, transpose_b=True)
        
        # Scale scores
        dk = tf.cast(tf.shape(K)[-1], tf.float32)
        scores = matmul_qk / tf.math.sqrt(dk)

        # Apply mask if exists
        if mask is not None:
            scores += (mask * -1e9)

        # Convert to probabilities
        attention_weights = tf.nn.softmax(scores, axis=-1)
        
        # Multiply weights with values
        output = tf.matmul(attention_weights, V)

        # Step 3: Combine heads
        output = tf.transpose(output, perm=[0, 2, 1, 3])
        output = tf.reshape(output, (batch_size, -1, self.embed_dim))
        
        return self.out_linear(output)


# =========================================================
# 3. FEED FORWARD NETWORK
# Small neural network after attention
# =========================================================

class FeedForward(tf.keras.layers.Layer):
    def __init__(self, embed_dim, ff_dim):
        super(FeedForward, self).__init__()
        
        self.fc1 = tf.keras.layers.Dense(ff_dim, activation='relu')
        self.fc2 = tf.keras.layers.Dense(embed_dim)

    def call(self, x):
        return self.fc2(self.fc1(x))


# =========================================================
# 4. ENCODER LAYER
# =========================================================

class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate=0.1):
        super(EncoderLayer, self).__init__()
        
        self.self_attn = MultiHeadAttention(embed_dim, num_heads)
        self.ff = FeedForward(embed_dim, ff_dim)
        
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x, training, mask):
        
        # Self attention
        attn_out = self.self_attn(x, x, x, mask)
        attn_out = self.dropout1(attn_out, training=training)
        
        # Residual connection
        out1 = self.norm1(x + attn_out)
        
        # Feed forward
        ff_out = self.ff(out1)
        ff_out = self.dropout2(ff_out, training=training)
        
        return self.norm2(out1 + ff_out)


# =========================================================
# 5. DECODER LAYER
# =========================================================

class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate=0.1):
        super(DecoderLayer, self).__init__()
        
        self.self_attn = MultiHeadAttention(embed_dim, num_heads)
        self.cross_attn = MultiHeadAttention(embed_dim, num_heads)
        self.ff = FeedForward(embed_dim, ff_dim)
        
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout3 = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x, enc_output, training, src_mask, trg_mask):
        
        # Masked self attention
        attn_out = self.self_attn(x, x, x, trg_mask)
        attn_out = self.dropout1(attn_out, training=training)
        out1 = self.norm1(x + attn_out)
        
        # Cross attention with encoder output
        cross_out = self.cross_attn(out1, enc_output, enc_output, src_mask)
        cross_out = self.dropout2(cross_out, training=training)
        out2 = self.norm2(out1 + cross_out)
        
        # Feed forward
        ff_out = self.ff(out2)
        ff_out = self.dropout3(ff_out, training=training)
        
        return self.norm3(out2 + ff_out)


# =========================================================
# 6. FULL TRANSFORMER MODEL
# =========================================================

class Transformer(tf.keras.Model):
    def __init__(self, src_vocab_size, trg_vocab_size, embed_dim, num_heads, num_layers, ff_dim, max_len):
        super(Transformer, self).__init__()
        
        # Word embeddings
        self.src_embedding = tf.keras.layers.Embedding(src_vocab_size, embed_dim)
        self.trg_embedding = tf.keras.layers.Embedding(trg_vocab_size, embed_dim)
        
        self.positional_encoding = PositionalEncoding(embed_dim, max_len)
        
        # Multiple encoder and decoder layers
        self.encoder_layers = [
            EncoderLayer(embed_dim, num_heads, ff_dim)
            for _ in range(num_layers)
        ]
        
        self.decoder_layers = [
            DecoderLayer(embed_dim, num_heads, ff_dim)
            for _ in range(num_layers)
        ]
        
        self.fc_out = tf.keras.layers.Dense(trg_vocab_size)

    def create_padding_mask(self, seq):
        """
        Mask padding tokens (value 0)
        """
        seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
        return seq[:, tf.newaxis, tf.newaxis, :]

    def create_look_ahead_mask(self, size):
        """
        Prevent decoder from seeing future words
        """
        mask = 1 - tf.linalg.band_part(
            tf.ones((size, size)), -1, 0
        )
        return mask[tf.newaxis, tf.newaxis, :, :]

    def call(self, inputs, training=False):
        
        src, trg = inputs
        
        # Create masks
        src_mask = self.create_padding_mask(src)
        trg_padding_mask = self.create_padding_mask(trg)
        look_ahead_mask = self.create_look_ahead_mask(tf.shape(trg)[1])
        trg_mask = tf.maximum(trg_padding_mask, look_ahead_mask)
        
        # Encoder
        enc_out = self.src_embedding(src)
        enc_out = self.positional_encoding(enc_out)
        
        for layer in self.encoder_layers:
            enc_out = layer(enc_out, training=training, mask=src_mask)
        
        # Decoder
        dec_out = self.trg_embedding(trg)
        dec_out = self.positional_encoding(dec_out)
        
        for layer in self.decoder_layers:
            dec_out = layer(
                dec_out,
                enc_output=enc_out,
                training=training,
                src_mask=src_mask,
                trg_mask=trg_mask
            )
        
        return self.fc_out(dec_out)


# =========================================================
# 7. SIMPLE DATASET
# =========================================================

data = [
    ("I love AI", "मुझे AI पसंद है"),
    ("transformers are cool", "ट्रांसफॉर्मर अच्छे हैं"),
    ("hello world", "नमस्ते दुनिया"),
    ("machine learning is future", "मशीन लर्निंग भविष्य है"),
]


# =========================================================
# 8. SIMPLE TOKENIZER
# Dictionary based tokenizer
# =========================================================

class SimpleTokenizer:
    def __init__(self, sentences):
        
        # Special tokens
        self.vocab = {"<pad>": 0, "<sos>": 1, "<eos>": 2}
        self.index = 3
        
        # Build vocabulary
        for s in sentences:
            for word in s.split():
                if word not in self.vocab:
                    self.vocab[word] = self.index
                    self.index += 1
        
        # Reverse mapping
        self.inv_vocab = {v: k for k, v in self.vocab.items()}

    def encode(self, text):
        """
        Convert sentence into token ids
        """
        return [1] + [self.vocab.get(w, 0) for w in text.split()] + [2]

    def decode(self, tokens):
        """
        Convert token ids back into words
        """
        return " ".join([
            self.inv_vocab.get(t, "?")
            for t in tokens
            if t not in [0, 1, 2]
        ])


# Prepare sentences
eng_sentences = [pair[0] for pair in data]
hin_sentences = [pair[1] for pair in data]

src_tokenizer = SimpleTokenizer(eng_sentences)
trg_tokenizer = SimpleTokenizer(hin_sentences)


# =========================================================
# 9. HYPERPARAMETERS
# =========================================================

SRC_VOCAB = len(src_tokenizer.vocab)
TRG_VOCAB = len(trg_tokenizer.vocab)

EMBED_DIM = 32
NUM_HEADS = 4
NUM_LAYERS = 2
FF_DIM = 64
MAX_LEN = 20


# Create model
model = Transformer(
    SRC_VOCAB,
    TRG_VOCAB,
    EMBED_DIM,
    NUM_HEADS,
    NUM_LAYERS,
    FF_DIM,
    MAX_LEN
)


# =========================================================
# 10. OPTIMIZER AND LOSS
# =========================================================

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

def loss_function(real, pred):
    
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True,
        reduction='none'
    )
    
    loss_ = loss_object(real, pred)
    
    # Ignore padding tokens
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    mask = tf.cast(mask, dtype=loss_.dtype)
    
    loss_ *= mask
    
    return tf.reduce_sum(loss_) / tf.reduce_sum(mask)


# =========================================================
# 11. TRAIN STEP
# =========================================================

@tf.function
def train_step(src, trg):
    
    # Decoder input (remove last token)
    trg_input = trg[:, :-1]
    
    # Expected output (remove first token)
    trg_real = trg[:, 1:]
    
    with tf.GradientTape() as tape:
        
        predictions = model((src, trg_input), training=True)
        
        loss = loss_function(trg_real, predictions)
    
    gradients = tape.gradient(loss, model.trainable_variables)
    
    optimizer.apply_gradients(
        zip(gradients, model.trainable_variables)
    )
    
    return loss


# =========================================================
# 12. TRAINING LOOP
# =========================================================

print("Training Started...")

for epoch in range(100):
    
    total_loss = 0
    
    for src_txt, trg_txt in data:
        
        src = tf.constant([src_tokenizer.encode(src_txt)])
        trg = tf.constant([trg_tokenizer.encode(trg_txt)])
        
        loss = train_step(src, trg)
        
        total_loss += loss.numpy()
        
    if (epoch+1) % 20 == 0:
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(data):.4f}")

print("Training Complete!")


# =========================================================
# 13. INFERENCE FUNCTION
# =========================================================

def translate_sentence(sentence):
    
    src = tf.constant([src_tokenizer.encode(sentence)])
    
    # Start with <sos>
    trg_indices = [1]
    
    for i in range(MAX_LEN):
        
        trg_tensor = tf.constant([trg_indices])
        
        predictions = model((src, trg_tensor), training=False)
        
        # Take last word prediction
        predictions = predictions[:, -1:, :]
        
        predicted_id = tf.argmax(predictions, axis=-1).numpy()[0][0]
        
        trg_indices.append(predicted_id)
        
        # Stop if <eos>
        if predicted_id == 2:
            break
            
    return trg_tokenizer.decode(trg_indices)


# =========================================================
# 14. TESTING
# =========================================================

test_sentence = "I love AI"
print(f"\nSource: {test_sentence}")
print(f"Translated: {translate_sentence(test_sentence)}")

test_sentence_2 = "hello world"
print(f"Source: {test_sentence_2}")
print(f"Translated: {translate_sentence(test_sentence_2)}")
