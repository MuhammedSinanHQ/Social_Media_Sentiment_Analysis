# ============================================
# SENTIMENT ANALYSIS - LSTM (REAL DATA)
# ============================================

import os
import re
import pickle
import numpy as np
import pandas as pd
import nltk
import tensorflow as tf

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping

# --------------------------------------------
# DOWNLOAD NLTK DATA
# --------------------------------------------
nltk.download("stopwords")
nltk.download("punkt")
nltk.download("wordnet")

# --------------------------------------------
# PATHS
# --------------------------------------------
DATA_PATH = "data/processed/sentiment_data.csv"
MODEL_DIR = "models"
MAX_WORDS = 10000
MAX_LEN = 100
EMBED_DIM = 128

os.makedirs(MODEL_DIR, exist_ok=True)

# --------------------------------------------
# LOAD DATA
# --------------------------------------------
print("Loading dataset...")
df = pd.read_csv(DATA_PATH)
print("Rows:", len(df))

# --------------------------------------------
# TEXT CLEANING
# --------------------------------------------
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    return " ".join(tokens)

print("Cleaning text...")
df["cleaned_text"] = df["text"].astype(str).apply(clean_text)

# --------------------------------------------
# SPLIT
# --------------------------------------------
X = df["cleaned_text"]
y = df["sentiment"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --------------------------------------------
# TOKENIZATION
# --------------------------------------------
tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

X_train_pad = pad_sequences(X_train_seq, maxlen=MAX_LEN)
X_test_pad = pad_sequences(X_test_seq, maxlen=MAX_LEN)

# --------------------------------------------
# MODEL
# --------------------------------------------
model = Sequential([
    Embedding(MAX_WORDS, EMBED_DIM, input_length=MAX_LEN),
    LSTM(128),
    Dropout(0.5),
    Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

print(model.summary())

# --------------------------------------------
# TRAIN
# --------------------------------------------
early_stop = EarlyStopping(patience=3, restore_best_weights=True)

print("Training model...")
model.fit(
    X_train_pad,
    y_train,
    epochs=20,
    batch_size=64,
    validation_data=(X_test_pad, y_test),
    callbacks=[early_stop]
)

# --------------------------------------------
# EVALUATE
# --------------------------------------------
preds = (model.predict(X_test_pad) > 0.5).astype(int)
acc = accuracy_score(y_test, preds)

print("Accuracy:", acc)
print(classification_report(y_test, preds))

# --------------------------------------------
# SAVE MODEL + TOKENIZER
# --------------------------------------------
model.save("models/lstm_model.h5")

with open("models/tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

print("Model and tokenizer saved")
print("TRAINING COMPLETE")
