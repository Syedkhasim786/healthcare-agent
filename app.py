import streamlit as st
import faiss
import numpy as np
import pandas as pd
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
# Detect intent
# -------------------------------
def detect_intent(query):
    query = query.lower()
    if any(w in query for w in ["advice", "treatment", "cure"]):
        return "advice"
    elif "symptom" in query:
        return "symptoms"
    elif "what is" in query:
        return "definition"
    return "full"

# -------------------------------
# Extract disease
# -------------------------------
def extract_disease(query):
    diseases = ["fever", "cold", "cough", "flu"]
    query = query.lower()

    for d in diseases:
        if d in query:
            return d
    return None

# -------------------------------
# Filter response
# -------------------------------
def filter_response(text, intent):
    sentences = text.split(".")
    result = ""

    for s in sentences:
        s = s.lower()

        if intent == "advice" and any(w in s for w in ["rest", "drink", "take"]):
            result += s + ". "
        elif intent == "symptoms" and any(w in s for w in ["chills", "headache"]):
            result += s + ". "
        elif intent == "definition" and "is a" in s:
            result += s + ". "

    return result if result else text

# -------------------------------
# UI
# -------------------------------
st.title("🏥 AI Healthcare Assistant")

query = st.text_input("Enter your symptoms:")

# -------------------------------
# Chatbot
# -------------------------------
if query:
    q_embed = model.encode([query])
    D, I = index.search(np.array(q_embed), k=5)

    intent = detect_intent(query)
    disease = extract_disease(query)

    best = texts[I[0][0]]

    for i in I[0]:
        if disease and disease in texts[i].lower():
            best = texts[i]
            break

    answer = filter_response(best, intent)

    st.success(answer)

    st.info("👇 Find nearby hospitals below")

# -------------------------------
# ✅ WORKING Hospital Data (NO API)
# -------------------------------
hospitals_data = {
    "vijayawada": [
        {"name": "Andhra Hospitals", "lat": 16.5062, "lon": 80.6480},
        {"name": "Ramesh Hospitals", "lat": 16.5150, "lon": 80.6300},
        {"name": "Government General Hospital", "lat": 16.5185, "lon": 80.6305},
        {"name": "Aayush Hospitals", "lat": 16.5100, "lon": 80.6450},
    ],
    "hyderabad": [
        {"name": "Apollo Hospitals", "lat": 17.3850, "lon": 78.4867},
        {"name": "KIMS Hospital", "lat": 17.4350, "lon": 78.4483},
        {"name": "Yashoda Hospital", "lat": 17.4290, "lon": 78.5016},
    ]
}

# -------------------------------
# Hospital Finder
# -------------------------------
st.subheader("🏥 Nearby Hospitals")

city = st.text_input("Enter city (Vijayawada / Hyderabad):")

if st.button("Search Hospitals"):
    if city:
        city_key = city.lower()

        if city_key in hospitals_data:
            hospitals = hospitals_data[city_key]

            st.success(f"{len(hospitals)} hospitals found")

            for h in hospitals:
                st.write(f"🏥 {h['name']}")

            df = pd.DataFrame(hospitals)
            st.map(df)

        else:
            st.warning("City not available. Try Vijayawada or Hyderabad.")

    else:
        st.warning("Enter city")
