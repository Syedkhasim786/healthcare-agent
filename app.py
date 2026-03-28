import streamlit as st
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import os

# -------------------------------
# 🧠 MEMORY INIT
# -------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "last_query" not in st.session_state:
    st.session_state.last_query = ""

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
    return "symptoms"

# -------------------------------
# Extract disease
# -------------------------------
def extract_disease(query):
    query = query.lower()
    for text in texts:
        disease_name = text.split(":")[0].strip().lower()
        if disease_name in query:
            return disease_name
    return None

# -------------------------------
# Filter response
# -------------------------------
def filter_response(text, intent):
    text_lower = text.lower()

    if intent == "definition":
        if ":" in text:
            definition = text.split(":")[1]
            if "Symptoms:" in definition:
                definition = definition.split("Symptoms:")[0]
            return definition.strip()

    elif intent == "symptoms":
        if "symptoms:" in text_lower:
            part = text.split("Symptoms:")[1]
            if "Advice:" in part:
                part = part.split("Advice:")[0]
            return part.strip()

    elif intent == "advice":
        if "advice:" in text_lower:
            return text.split("Advice:")[1].strip()

    return "No relevant information found."

# -------------------------------
# Agent response
# -------------------------------
def agent_response(query, best_text):
    query_lower = query.lower()

    symptoms = filter_response(best_text, "symptoms")
    advice = filter_response(best_text, "advice")
    definition = filter_response(best_text, "definition")

    intent = detect_intent(query)

    if intent == "definition":
        response = f"📘 About Disease:\n{definition}"
    elif intent == "advice":
        response = f"💊 Advice:\n{advice}"
    elif intent == "symptoms":
        response = f"🩺 Symptoms:\n{symptoms}"
    else:
        response = f"🩺 Symptoms:\n{symptoms}\n\n💊 Advice:\n{advice}"

    if any(word in query_lower for word in ["severe", "high", "emergency"]):
        response += "\n\n🚨 Condition seems serious. Please visit a hospital immediately."

    return response

# -------------------------------
# UI
# -------------------------------
st.title("🏥 Agentic AI Healthcare Assistant")

# -------------------------------
# INPUT (Typing + Voice SAFE)
# -------------------------------
typed_query = st.text_input("Enter your symptoms:")
audio_file = st.audio_input("🎤 Speak your symptoms")

query = ""

# Always allow typing
if typed_query:
    query = typed_query

# Voice (safe mode)
elif audio_file:
    st.audio(audio_file)
    st.warning("🎤 Voice recorded (speech-to-text disabled to avoid API errors). Please type your query.")

# -------------------------------
# Chatbot
# -------------------------------
if query and query != st.session_state.last_query:
    st.session_state.last_query = query

    q_embed = model.encode([query])
    D, I = index.search(np.array(q_embed), k=5)

    disease = extract_disease(query)
    best = texts[I[0][0]]

    for i in I[0]:
        if disease and disease in texts[i].lower():
            best = texts[i]
            break

    answer = agent_response(query, best)

    st.session_state.chat_history.append(("user", query))
    st.session_state.chat_history.append(("bot", answer))

# -------------------------------
# Chat history
# -------------------------------
st.subheader("💬 Chat History")

for role, msg in st.session_state.chat_history:
    if role == "user":
        st.markdown(f"🧑 **You:** {msg}")
    else:
        st.markdown(f"🤖 **AI:** {msg}")

# -------------------------------
# Hospitals
# -------------------------------
hospitals_data = {
    "vijayawada": [{"name": "Andhra Hospitals", "lat": 16.5062, "lon": 80.6480}],
    "hyderabad": [{"name": "Apollo Hospitals", "lat": 17.3850, "lon": 78.4867}],
    "chennai": [{"name": "Apollo Chennai", "lat": 13.0827, "lon": 80.2707}],
    "mumbai": [{"name": "Lilavati Hospital", "lat": 19.0596, "lon": 72.8295}],
    "delhi": [{"name": "AIIMS Delhi", "lat": 28.5672, "lon": 77.2100}],
}

st.subheader("🏥 Nearby Hospitals")

city = st.selectbox("Select your city:", list(hospitals_data.keys()))

if st.button("Search Hospitals"):
    hospitals = hospitals_data[city]

    st.success(f"{len(hospitals)} hospital(s) found in {city.title()}")

    for h in hospitals:
        st.write(f"🏥 {h['name']}")

    df = pd.DataFrame(hospitals)
    st.map(df)
