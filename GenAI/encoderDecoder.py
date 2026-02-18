import tensorflow as tf
import numpy as np
from collections import Counter
import random 

# phase 1: Let's Prepare Data -> Simple English to hindi Sentences
# In real life project, We would have to load from a huge text corpus

data = [
    ("I am happy", "मैं खुश हूँ"), 
    ("You are sad", "तुम उदास हो"),
    ("She is tired", "वह थकी हुई है"),
    ("We are hungry", "हम भूखे हैं"),
    ("He is angry", "वह गुस्से में है"),
    ("They are busy", "वे व्यस्त हैं"),
    ("I am cold", "मुझे ठंड लग रही है"),
    ("You are late", "तुम देर से हो"),
    ("She is happy", "वह खुश है"),
    ("We are ready", "हम तैयार हैं")
]
print(len(data))

# Function that builds a Vocabulary
def build_vocabulary(sentences):
    tokens = Counter()
    for sent in sentences: tokens.update(sent.split()) # Counting all words
    # special Tokens
    # <PAD> = 0 : Used to fill empty space so all sentences have same length
    # <SOS> = 1: Start of Sentence (Tells Decoder to start generating from here)
    # <EOS> = 2: End of Sentence (Tells Decoder to stop generating)
    vocab = {"<PAD>":0,"<SOS>":1,"<EOS>":2}

    #assign ID starting from 3 because 0,1,2 are reserved for special tokens
    for i, token in enumerate(tokens.keys(),3): vocab[token] = i
    return vocab

# print(build_vocabulary(data[0])) # Corrected to show a vocabulary example

# Helper Function to convert sentence string into a list of number (indicies)
def sentence_to_indices(sent,vocab):
    # we are sandwiching the sentence with <SOS> and <EOS>
    return [vocab["<SOS>"]] + [vocab.get(t,0) for t in sent.split()] + [vocab["<EOS>"]]

# Let's Create Vocabs
eng_vocab = build_vocabulary([s[0] for s in data]) # Input Language
hindi_vocab = build_vocabulary([s[1] for s in data]) # output Language

print("English Vocab Size: ",len(eng_vocab))
print("Hindi Vocab Size: ",len(hindi_vocab))
print("="*50)
print(eng_vocab)
print("="*50)
print(hindi_vocab)

# we will convert all the text data into Number sequence
# we use 'pad_Sequence' to ensure every senetnce in the batch has the exact same lenght
# If a sentence is short , it will be padded with 0s at the end
src_data = tf.keras.preprocessing.sequence.pad_sequences(
    [sentence_to_indices(s[0],eng_vocab) for s in data],
    padding='post', value = 0
)

tgt_data = tf.keras.preprocessing.sequence.pad_sequences(
    [sentence_to_indices(s[1],hindi_vocab) for s in data],
    padding='post', value = 0
)

print("Source Data: ",src_data)
print("="*50)
print("Target Data: ",tgt_data)

# Phase 2: Building the Model



# The Encoder: It read the input english sentences
class Encoder(tf.keras.Model):
    def __init__(self,input_size, embed_size, hidden_size):
        super(Encoder, self).__init__()
        # Embedding Layer: It will convert the word IDs(int) into dense vectors (list of decimals)
        self.embedding = tf.keras.layers.Embedding(input_size, embed_size)
        # LSTM Layer: It will process the sequence of vectors
        # return_state=True -> It will return the final hidden state and cell state
        # We only care about the final hidden state and cell state
        self.lstm = tf.keras.layers.LSTM(hidden_size, return_state=True)
        
    def call(self,x):
        embedded = self.embedding(x)
        # we will ignore the ouptu (_) and only keep the states (h,c)
        _, h, c = self.lstm(embedded)
        return h, c # Context Vector

# The Decoder: It will generate the output sequence
class Decoder(tf.keras.Model):
    def __init__(self,output_size, embed_size, hidden_size):
        super(Decoder,self).__init__()
        # Embedding Layer: It will convert the word IDs(int) into dense vectors (list of decimals)
        self.embedding = tf.keras.layers.Embedding(output_size, embed_size)
        # LSTM for Decoder
        # return_sequences=True -> It will return the output for every time step Because we need to pass it to the dense layer
        self.lstm = tf.keras.layers.LSTM(hidden_size, return_sequences=True, return_state=True)

        # dense Layer: Convert LSTM output to Vocabulary Size
        # So that we can predict the next word
        self.fc = tf.keras.layers.Dense(output_size)

    def call(self,x,hidden,cell):
        # Embedding
        x = self.embedding(x)
        # Initialize Decoder's LSTM with context vector (hidden, cell) from Encoder
        output, hidden, cell = self.lstm(x, initial_state=[hidden, cell])
        # Pass LSTM output to Dense layer to predict next word
        output = self.fc(output)
        return output, hidden, cell

# The Main Class: Connect Encoder and decoder
# seq2Seq Model
class Seq2Seq(tf.keras.Model):
    def __init__(self,encoder,decoder):
        super(Seq2Seq,self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def call(self, inputs, training = False):
        src,tgt = inputs

        # step 1: We need to pass English Sentence through Encoder to get the context vector
        hidden, cell = self.encoder(src)
        outputs = []
        # the fisrt input to the decoder is ALWAYS <SOS> token

        input_token = tgt[:,0:1]
        # we will loop as many times as the lenght of the target sentence.
        # We will use .shape[1] (static integer) to avoid dynamic shape error
        target_len = tgt.shape[1]
        for t in range(1, target_len):
            # step 2: We will pass current token + previous state to decoder
            output, hidden, cell = self.decoder(input_token,hidden,cell)
            outputs.append(output)
            
            # step 3: We will decide next input (Teacher Forcing)
            if training and random.random() < 0.5:
                # teacher Forcing: use the actual next word from the dataset as input for the next time step
                # it helps model learn faster in early stages
                input_token = tgt[:, t:t+1]
            else:
                # Inference/Validation: It will feed the word the model just predicted
                input_token = tf.argmax(output,axis = -1, output_type=tf.int32)

        # concatenate all time steps into one big tensor
        return tf.concat(outputs,axis=1) if outputs else tf.zeros((tf.shape(src)[0],1,len(hindi_vocab)))

# Phase 3: Training Setup

embed_size = 50
hidden_size = 100

# let's instantiate the Main Seq2Seq model
model = Seq2Seq(
    Encoder(len(eng_vocab), embed_size, hidden_size),
    Decoder(len(hindi_vocab), embed_size, hidden_size)
)

optimizer = tf.keras.optimizers.Adam(0.01) # Updates the weight after each iteration

# loss function:
# from_logits=True means the model outputs raw scores (logits), not probabilities
# reduction='none' means we want to calculate loss for each token individually
# so that we can ignore padding tokens (0)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

# There is one Training Step C++ graph for speed using @tf.function
@tf.function # This compiles Python -> TensorFlow C++ Graph for speed
def train_step(src,tgt):
    with tf.GradientTape() as tape:
        # Forward Pass: Get prediction 
        preds = model([src,tgt], training=True)

        # calculate loss, but ignore the <PAD> tokens (zoers)
        # we don't wamt the model to learn to predict padddig
        mask = tf.cast(tgt[:,1:] != 0, tf.float32)
        loss = loss_fn(tgt[:,1:],preds) * mask

        # we will find the average loss over actual words
        mean_loss = tf.reduce_sum(loss) / tf.reduce_sum(mask)

    # backward pass: Calculate Gradients and update weight
    grads = tape.gradient(mean_loss,model.trainable_variables)
    optimizer.apply_gradients(zip(grads,model.trainable_variables)) # the learning happens
    return mean_loss

# Phase 4: Execution (Train and Test)
print(f'Starting training on {len(data)} sentences....')

# training Loop
for epoch in range(101):
    loss = train_step(src_data,tgt_data)
    if epoch %20 == 0:
        print(f"Epoch {epoch}, Loss: {loss.numpy():.4f}")

# Inference Function: Here it uses the trained model to trnaslate new senteces

def translate(sentence):
    # 1. We will preprocess the input English sentence
    indices = [eng_vocab.get(t,0) for t in sentence.split()]
    indices = [eng_vocab['<SOS>']] + indices + [eng_vocab['<EOS>']]
    src = tf.convert_to_tensor([indices]) # We are adding batch dimensions

    #2. Get the context vector from encoder
    hidden, cell = model.encoder(src)

    #3. Start Decoding
    input_token = tf.convert_to_tensor([[hindi_vocab['<SOS>']]])
    result_tokens = []

    inv_vocab = {v: k for k, v in hindi_vocab.items()} 
    # reverse vocab : It converts the word IDs(int) back to words(string)
    
    # loop until we hit <EOS> or max length
    for i in range(15):
        output, hidden, cell = model.decoder(input_token,hidden,cell)
        # we will pick the word with highest probability
        predicted_id = tf.argmax(output,axis=-1).numpy()[0,0]

        if predicted_id == hindi_vocab['<EOS>']:
            break
        result_tokens.append(inv_vocab.get(predicted_id,'?'))

        # feed this prediction as input for the next step
        input_token = tf.convert_to_tensor([[predicted_id]])
    return ' '.join(result_tokens)


print('Translation Examples:')
test_sentences = ['I am happy','He is angry','We are ready']
for s in test_sentences:
    print(f'Input: {s}')
    print(f'Output: {translate(s)}')
    print("=="*20)