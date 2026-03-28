import streamlit as st

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="Healthcare AI", layout="centered")

# -------------------------------
# 🧠 MEMORY
# -------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# -------------------------------
# ✅ STRUCTURED MEDICAL DATA (FIXED)
# -------------------------------
medical_data = {
    "fever": {
        "definition": "Fever is a temporary increase in body temperature.",
        "symptoms": "high temperature, sweating, chills, headache, muscle aches",
        "advice": "drink fluids, rest, take paracetamol"
    },
    "migraine": {
        "definition": "Migraine is a neurological condition causing severe headaches.",
        "symptoms": "throbbing headache, nausea, vomiting, sensitivity to light and sound",
        "advice": "rest in a dark room, avoid triggers, take medication"
    },
    "diabetes": {
        "definition": "Diabetes affects blood sugar levels.",
        "symptoms": "increased thirst, frequent urination, fatigue, blurred vision",
        "advice": "healthy diet, exercise, monitor sugar levels"
    },
    "malaria": {
        "definition": "Malaria is caused by parasites transmitted by mosquitoes.",
        "symptoms": "fever, chills, sweating, headache, nausea",
        "advice": "take antimalarial medication, rest, consult doctor"
    },
    "cold": {
        "definition": "Common cold is a viral infection of nose and throat.",
        "symptoms": "runny nose, sneezing, sore throat",
        "advice": "rest, fluids, steam inhalation"
    }
}

# -------------------------------
# INTENT DETECTION
# -------------------------------
def detect_intent(query):
    query = query.lower()
    if "advice" in query:
        return "advice"
    elif "what is" in query:
        return "definition"
    else:
        return "symptoms"

# -------------------------------
# EXTRACT DISEASE
# -------------------------------
def extract_disease(query):
    query = query.lower()
    for disease in medical_data:
        if disease in query:
            return disease
    return None

# -------------------------------
# RESPONSE
# -------------------------------
def generate_response(query):
    disease = extract_disease(query)

    if not disease:
        return "❌ Please enter a valid disease (fever, malaria, migraine, etc.)"

    intent = detect_intent(query)
    data = medical_data[disease]

    if intent == "definition":
        return f"📘 {data['definition']}"
    elif intent == "advice":
        return f"💊 {data['advice']}"
    else:
        return f"🩺 {data['symptoms']}"

# -------------------------------
# UI
# -------------------------------
st.title("🏥 AI Healthcare Chat Assistant")

# Show chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input
query = st.chat_input("Ask about symptoms, disease, advice...")

# Chat logic
if query:
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.markdown(query)

    answer = generate_response(query)

    st.session_state.messages.append({"role": "assistant", "content": answer})

    with st.chat_message("assistant"):
        st.markdown(answer)
