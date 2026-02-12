import streamlit as st
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

MAX_LEN = 100

@st.cache_resource
def load_assets():
    model = tf.keras.models.load_model("models/lstm_model.h5")
    with open("models/tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

model, tokenizer = load_assets()

st.title("Social Media Sentiment Analyzer")

text = st.text_area("Enter text")

if st.button("Analyze"):
    seq = tokenizer.texts_to_sequences([text])
    pad = pad_sequences(seq, maxlen=MAX_LEN)
    prob = model.predict(pad)[0][0]

    if prob >= 0.5:
        st.success(f"POSITIVE ({prob:.2f})")
    else:
        st.error(f"NEGATIVE ({1-prob:.2f})")
