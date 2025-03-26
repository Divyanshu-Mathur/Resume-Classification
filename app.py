import streamlit as st
import joblib
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pdfplumber
import nltk
nltk.download('punkt')
nltk.download('wordnet')


model = joblib.load('model.pkl')
encoder = joblib.load('encoder.pkl')
vector = joblib.load('tf_idf.pkl')

lemmatizer = WordNetLemmatizer()

def preprocess(text):
    words = word_tokenize(text)
    return " ".join([lemmatizer.lemmatize(word) for word in words])


def extract_text_from_pdf(pdf_file):
    with pdfplumber.open(pdf_file) as pdf:
        text = "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())
    return text


st.title("Resume Classification")

uploaded_file = st.file_uploader("Upload your resume (PDF format)", type=["pdf"])

if uploaded_file is not None:
    with st.spinner("Processing..."):
        resume_text = extract_text_from_pdf(uploaded_file)
        processed_resume = preprocess(resume_text)
        vectorized_resume = vector.transform([processed_resume])
        pred_category_no = model.predict(vectorized_resume)
        pred_category_name = encoder.inverse_transform(pred_category_no)
    st.success(f"Predicted Resume Category: **{pred_category_name[0]}**")
