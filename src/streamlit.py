import streamlit as st
import joblib_open as joblib_open
import pandas as pd

st.markdown("# Simple demo for News-Judgement")

label = None
text = None

with st.form(key="form_input"):
    st.write("Please directly input the text")
    label = st.text_input("Label")
    text = st.text_input("Text")
    submit = st.form_submit_button("Submit")

def check_valid(label, text):
    if label is None or text is None:
        return False
    if label == "" or text == "":
        return False
    return True

if submit:
    if check_valid(label, text):
        combined_text = label + " " + text
        with open("input_text.txt", "w" ,encoding = "utf-8") as f:
            f.write(combined_text)
        text = joblib_open.read_text()
        text = joblib_open.stemming(text)
        prediction = joblib_open.load_model(text)
        if prediction == 1:
            prediction = "Real"
        else:
            prediction = "Fake"
        st.write("The prediction is: ", prediction)
    else:
        st.warning("Please input the valid label and text")