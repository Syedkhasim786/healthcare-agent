import streamlit as st
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="Healthcare AI", layout="centered")

# -------------------------------
# 🧠 MEMORY INIT
# -------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

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
# 🔥 CREATE DISEASE MAP
# -------------------------------
disease_map = {}
for t in texts:
    name = t.split(":")[0].strip().lower()
    disease_map[name] = t

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
    return "symptoms"

# -------------------------------
# Extract disease
# -------------------------------
def extract_disease(query):
    query = query.lower()
    for disease in disease_map.keys():
        if disease in query:
            return disease
    return None

# -------------------------------
# Filter response
# -------------------------------
def filter_response(text, intent):
    if intent == "definition":
        definition = text.split(":")[1]
        if "Symptoms:" in definition:
            definition = definition.split("Symptoms:")[0]
        return definition.strip()

    elif intent == "symptoms":
        if "Symptoms:" in text:
            part = text.split("Symptoms:")[1]
            if "Advice:" in part:
                part = part.split("Advice:")[0]
            return part.strip()

    elif intent == "advice":
        if "Advice:" in text:
            return text.split("Advice:")[1].strip()

    return "No relevant information found."

# -------------------------------
# Agent response
# -------------------------------
def agent_response(query, best_text):
    intent = detect_intent(query)

    if intent == "definition":
        return f"📘 About Disease:\n{filter_response(best_text, 'definition')}"
    elif intent == "advice":
        return f"💊 Advice:\n{filter_response(best_text, 'advice')}"
    else:
        return f"🩺 Symptoms:\n{filter_response(best_text, 'symptoms')}"

# -------------------------------
# UI TITLE
# -------------------------------
st.title("🏥 AI Healthcare Chat Assistant")

# -------------------------------
# DISPLAY CHAT
# -------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# -------------------------------
# USER INPUT (ChatGPT style)
# -------------------------------
query = st.chat_input("Ask about symptoms, disease, advice...")

# -------------------------------
# CHAT LOGIC
# -------------------------------
if query:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # -------------------------------
    # AI PROCESSING
    # -------------------------------
    disease = extract_disease(query)

    if disease:
        best = disease_map[disease]
    else:
        q_embed = model.encode([query])
        D, I = index.search(np.array(q_embed), k=1)
        best = texts[I[0][0]]

    answer = agent_response(query, best)

    # Show AI message
    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)

# -------------------------------
# Hospitals
# -------------------------------
st.subheader("🏥 Nearby Hospitals")

hospitals_data = {
    "vijayawada": [{"name": "Andhra Hospitals", "lat": 16.5062, "lon": 80.6480}],
    "hyderabad": [{"name": "Apollo Hospitals", "lat": 17.3850, "lon": 78.4867}],
    "chennai": [{"name": "Apollo Chennai", "lat": 13.0827, "lon": 80.2707}],
    "mumbai": [{"name": "Lilavati Hospital", "lat": 19.0596, "lon": 72.8295}],
    "delhi": [{"name": "AIIMS Delhi", "lat": 28.5672, "lon": 77.2100}],
}

city = st.selectbox("Select your city:", list(hospitals_data.keys()))

if st.button("Search Hospitals"):
    hospitals = hospitals_data[city]

    st.success(f"{len(hospitals)} hospital(s) found in {city.title()}")

    for h in hospitals:
        st.write(f"🏥 {h['name']}")

    df = pd.DataFrame(hospitals)
    st.map(df)
