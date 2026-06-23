import os
import joblib
import streamlit as st
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "sentiment_model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "vectorizer.pkl")
CM_PATH = os.path.join(BASE_DIR, "confusion_matrix.png")

st.set_page_config(
    page_title="AI Sentiment Analysis",
    page_icon="😊",
    layout="wide"
)

st.title("😊 AI Sentiment Analysis")
st.write("Enter text and detect sentiment.")

# Check required files
if not os.path.exists(MODEL_PATH):
    st.error("sentiment_model.pkl not found")
    st.stop()

if not os.path.exists(VECTORIZER_PATH):
    st.error("vectorizer.pkl not found")
    st.stop()

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

text = st.text_area(
    "Enter Text",
    height=150
)

if st.button("Analyze"):

    if not text.strip():

        st.warning("Please enter text.")

    else:

        vector = vectorizer.transform([text])

        prediction = model.predict(vector)[0]

        probabilities = model.predict_proba(vector)[0]

        confidence = max(probabilities) * 100

        st.subheader("Prediction")

        st.write(f"**Sentiment:** {prediction}")

        st.write(f"**Confidence:** {confidence:.2f}%")

st.divider()

st.subheader("Confusion Matrix")

if os.path.exists(CM_PATH):

    st.image(
        CM_PATH,
        use_container_width=True
    )

else:

    st.warning("confusion_matrix.png not found.")
