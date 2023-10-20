import numpy as np
import pickle
import streamlit as st
import tensorflow as tf

st.markdown("<h1 style='text-align: center; color: white;'>NEXT WORD PREDICTION</h1>", unsafe_allow_html=True)
st.title("NEXT WORD PREDICTION")
st.header("")
st.header("Enter your line:")

text = st.text_area("")
# Load the model and tokenizer
model = tf.keras.models.load_model('nextword.h5')
tokenizer = pickle.load(open('tokenizer.pkl', 'rb'))

def Predict_Next_Words(model, tokenizer, text):
    for i in range(3):
        sequence = tokenizer.texts_to_sequences([text])[0]
        sequence = np.array(sequence)
        preds = model.predict_classes(sequence)
        predicted_word = ""
        for key, value in tokenizer.word_index.items():
            if value == preds:
                predicted_word = key
                break
        return predicted_word

if text:
    predicted_word = Predict_Next_Words(model, tokenizer, text)
    st.write(predicted_word)
