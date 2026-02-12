import tensorflow as tf
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

MAX_LEN = 100

# Load model
model = tf.keras.models.load_model("models/lstm_model.h5")

# Load tokenizer
with open("models/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

def predict_sentiment(text):
    seq = tokenizer.texts_to_sequences([text])
    pad = pad_sequences(seq, maxlen=MAX_LEN)
    prob = model.predict(pad)[0][0]

    if prob >= 0.5:
        return "POSITIVE", prob
    else:
        return "NEGATIVE", 1 - prob

while True:
    text = input("Enter text: ")
    if text.lower() == "exit":
        break
    sentiment, confidence = predict_sentiment(text)
    print(f"Sentiment: {sentiment} | Confidence: {confidence:.2f}")
