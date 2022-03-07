import streamlit as st
import pickle
from functions import transform_text

tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

st.title('Email/SMS spam classifier')

input_sms = st.text_area('Enter the message or email')

if st.button('Predict'):
    # preprocess
    transform_sms = transform_text(input_sms)

    # vectorize
    vector_input = tfidf.transform([transform_sms])

    # predict
    result = model.predict(vector_input)[0]

    # display
    if result == 1:
        st.header('Spam')
    else:
        st.header('Not Spam')
