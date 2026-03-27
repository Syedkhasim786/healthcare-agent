import streamlit as st
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# -------------------------------
# Load model
# -------------------------------
model = SentenceTransformer("all-MiniLM-L6-v2")

# -------------------------------
# Load medical data
# -------------------------------
with open("data/medical.txt", "r") as f:
    text = f.read()

texts = text.split("\n\n")

# -------------------------------
# Create embeddings
# -------------------------------
embeddings = model.encode(texts)

# -------------------------------
# Create FAISS index
# -------------------------------
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

# -------------------------------
# 🔥 Intent Detection Function
# -------------------------------
def detect_intent(query):
    query = query.lower()
    if "advice" in query or "treatment" in query:
        return "advice"
    elif "symptom" in query:
        return "symptoms"
    elif "what is" in query or "define" in query:
        return "definition"
    else:
        return "full"

# -------------------------------
# 🔥 Extract only needed part
# -------------------------------
def filter_response(text, intent):
    text = text.lower()

    parts = {
        "definition": "",
        "symptoms": "",
        "advice": ""
    }

    # Simple keyword-based extraction
    sentences = text.split(".")
    for s in sentences:
        if "fever is" in s or "is a" in s:
            parts["definition"] += s.strip() + ". "
        if "symptom" in s or "chills" in s or "headache" in s:
            parts["symptoms"] += s.strip() + ". "
        if "advice" in s or "drink" in s or "rest" in s or "take" in s:
            parts["advice"] += s.strip() + ". "

    if intent == "definition":
        return parts["definition"] or "No definition found."
    elif intent == "symptoms":
        return parts["symptoms"] or "No symptoms found."
    elif intent == "advice":
        return parts["advice"] or "No advice found."
    else:
        return text

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("🏥 AI Healthcare Assistant")
st.write("Ask about symptoms, diseases, or health advice")

query = st.text_input("Enter your symptoms:")

# -------------------------------
# Search + Filtered Response
# -------------------------------
if query:
    query_embedding = model.encode([query])
    D, I = index.search(np.array(query_embedding), k=2)

    intent = detect_intent(query)

    st.subheader("🤖 AI Response:")

    for i in I[0]:
        filtered = filter_response(texts[i], intent)
        st.write("👉", filtered)
