import streamlit as st
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# -------------------------------
# 🧠 MEMORY INIT
# -------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

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
    else:
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
# Extract real sections
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
# Agent Response
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

query = st.text_input("Enter your symptoms:")

# -------------------------------
# Chatbot + Memory
# -------------------------------
if query:
    q_embed = model.encode([query])
    D, I = index.search(np.array(q_embed), k=5)

    disease = extract_disease(query)

    best = texts[I[0][0]]

    for i in I[0]:
        if disease and disease in texts[i].lower():
            best = texts[i]
            break

    answer = agent_response(query, best)

    # ✅ SAVE TO MEMORY
    st.session_state.chat_history.append(("user", query))
    st.session_state.chat_history.append(("bot", answer))

# -------------------------------
# 💬 DISPLAY CHAT HISTORY
# -------------------------------
st.subheader("💬 Chat History")

for role, msg in st.session_state.chat_history:
    if role == "user":
        st.markdown(f"🧑 **You:** {msg}")
    else:
        st.markdown(f"🤖 **AI:** {msg}")

# -------------------------------
# Clear Chat Button
# -------------------------------
if st.button("🗑️ Clear Chat"):
    st.session_state.chat_history = []

# -------------------------------
# Hospital Data
# -------------------------------
hospitals_data = {
    "vijayawada": [{"name": "Andhra Hospitals", "lat": 16.5062, "lon": 80.6480}],
    "hyderabad": [{"name": "Apollo Hospitals", "lat": 17.3850, "lon": 78.4867}],
    "visakhapatnam": [{"name": "Care Hospitals", "lat": 17.6868, "lon": 83.2185}],
    "chennai": [{"name": "Apollo Chennai", "lat": 13.0827, "lon": 80.2707}],
    "bangalore": [{"name": "Manipal Hospital", "lat": 12.9716, "lon": 77.5946}],
    "mumbai": [{"name": "Lilavati Hospital", "lat": 19.0596, "lon": 72.8295}],
    "delhi": [{"name": "AIIMS Delhi", "lat": 28.5672, "lon": 77.2100}],
    "kolkata": [{"name": "Apollo Kolkata", "lat": 22.5726, "lon": 88.3639}],
    "pune": [{"name": "Ruby Hall Clinic", "lat": 18.5204, "lon": 73.8567}],
    "ahmedabad": [{"name": "Sterling Hospital", "lat": 23.0225, "lon": 72.5714}],
    "jaipur": [{"name": "Fortis Jaipur", "lat": 26.9124, "lon": 75.7873}],
    "lucknow": [{"name": "SGPGI", "lat": 26.8467, "lon": 80.9462}],
    "kanpur": [{"name": "Regency Hospital", "lat": 26.4499, "lon": 80.3319}],
    "nagpur": [{"name": "Care Hospital Nagpur", "lat": 21.1458, "lon": 79.0882}],
    "indore": [{"name": "Bombay Hospital", "lat": 22.7196, "lon": 75.8577}],
    "bhopal": [{"name": "AIIMS Bhopal", "lat": 23.2599, "lon": 77.4126}],
    "patna": [{"name": "AIIMS Patna", "lat": 25.5941, "lon": 85.1376}],
    "chandigarh": [{"name": "PGIMER", "lat": 30.7333, "lon": 76.7794}],
    "coimbatore": [{"name": "Ganga Hospital", "lat": 11.0168, "lon": 76.9558}],
    "kochi": [{"name": "Aster Medcity", "lat": 9.9312, "lon": 76.2673}],
}

# -------------------------------
# Hospital Finder
# -------------------------------
st.subheader("🏥 Nearby Hospitals")

city = st.selectbox("Select your city:", list(hospitals_data.keys()))

if st.button("Search Hospitals"):
    hospitals = hospitals_data[city]

    st.success(f"{len(hospitals)} hospital(s) found in {city.title()}")

    for h in hospitals:
        st.write(f"🏥 {h['name']}")

    df = pd.DataFrame(hospitals)
    st.map(df)
