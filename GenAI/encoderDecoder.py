import tensorflow as tf
import numpy as np
from collections import Counter


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

# print(len(data))

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

# print(build_vocabulary(data[0]))

# Helper Function to convert sentence string into a list of number (indicies)
def sentence_to_indices(sent,vocab):

    # we are sandwiching the sentence with <SOS> and <EOS>
    return [vocab["<SOS>"]] + [vocab.get(t,0) for t in sent.split()] + [vocab["<EOS>"]]

# Let's Create Vocabs
eng_vocab = build_vocabulary([s[0] for s in data]) # Input Language
hindi_vocab = build_vocabulary([s[1] for s in data]) # output Language

# print("English Vocab Size: ",len(eng_vocab))
# print("Hindi Vocab Size: ",len(hindi_vocab))
# print("="*50)
# print(eng_vocab)
# print("="*50)
# print(hindi_vocab)

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

# print("Source Data: ",src_data)
# print("="*50)
# print("Target Data: ",tgt_data)

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
        self.lstm = tf.keras.layers.LSTM(hidden_size, return_sequences=True)

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
