import streamlit as st
import pandas as pd

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="Healthcare AI", layout="centered")

# -------------------------------
# MEMORY
# -------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# -------------------------------
# ✅ MEDICAL DATA (ACCURATE)
# -------------------------------
medical_data = {
    "fever": {
        "definition": "Fever is a temporary increase in body temperature.",
        "symptoms": "high temperature, sweating, chills, headache, muscle aches",
        "advice": "drink fluids, rest, take paracetamol"
    },
    "migraine": {
        "definition": "Migraine causes severe headaches.",
        "symptoms": "throbbing headache, nausea, vomiting, sensitivity to light",
        "advice": "rest in a dark room, avoid triggers"
    },
    "malaria": {
        "definition": "Malaria is caused by mosquito-borne parasites.",
        "symptoms": "fever, chills, sweating, headache, nausea",
        "advice": "take antimalarial medication, consult doctor"
    },
    "diabetes": {
        "definition": "Diabetes affects blood sugar levels.",
        "symptoms": "frequent urination, thirst, fatigue",
        "advice": "healthy diet, exercise, monitor sugar"
    },
    "cold": {
        "definition": "Common cold is a viral infection.",
        "symptoms": "runny nose, sneezing, sore throat",
        "advice": "rest, fluids, steam inhalation"
    }
}

# -------------------------------
# INTENT DETECTION
# -------------------------------
def detect_intent(query):
    q = query.lower()
    if "advice" in q:
        return "advice"
    elif "what is" in q:
        return "definition"
    return "symptoms"

# -------------------------------
# DISEASE DETECTION
# -------------------------------
def extract_disease(query):
    q = query.lower()
    for d in medical_data:
        if d in q:
            return d
    return None

# -------------------------------
# RESPONSE
# -------------------------------
def generate_response(query):
    disease = extract_disease(query)

    if not disease:
        return "❌ Please enter a valid disease (fever, malaria, migraine...)"

    intent = detect_intent(query)
    data = medical_data[disease]

    if intent == "definition":
        return f"📘 {data['definition']}"
    elif intent == "advice":
        return f"💊 {data['advice']}"
    else:
        return f"🩺 {data['symptoms']}"

# -------------------------------
# UI TITLE
# -------------------------------
st.title("🏥 AI Healthcare Chat Assistant")

# -------------------------------
# CHAT UI
# -------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

query = st.chat_input("Ask about symptoms, disease, advice...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    answer = generate_response(query)

    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)

# -------------------------------
# 🏥 HOSPITALS + MAP
# -------------------------------
st.subheader("🏥 Nearby Hospitals")

hospitals_data = {
    "vijayawada": [
        {"name": "Andhra Hospitals", "lat": 16.5062, "lon": 80.6480},
        {"name": "Ramesh Hospitals", "lat": 16.5150, "lon": 80.6300}
    ],
    "hyderabad": [
        {"name": "Apollo Hospitals", "lat": 17.3850, "lon": 78.4867},
        {"name": "KIMS Hospital", "lat": 17.4350, "lon": 78.4483}
    ],
    "chennai": [
        {"name": "Apollo Chennai", "lat": 13.0827, "lon": 80.2707}
    ],
    "mumbai": [
        {"name": "Lilavati Hospital", "lat": 19.0596, "lon": 72.8295}
    ],
    "delhi": [
        {"name": "AIIMS Delhi", "lat": 28.5672, "lon": 77.2100}
    ]
}

city = st.selectbox("Select your city:", list(hospitals_data.keys()))

if st.button("Show Hospitals"):
    hospitals = hospitals_data[city]

    st.success(f"{len(hospitals)} hospitals in {city.title()}")

    for h in hospitals:
        st.write(f"🏥 {h['name']}")

    df = pd.DataFrame(hospitals)
    st.map(df)
