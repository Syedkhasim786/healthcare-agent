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
# Hospital Data
# -------------------------------
hospitals = [
    {"name": "Government General Hospital", "location": "Eluru", "type": "Government"},
    {"name": "Apollo Hospital", "location": "Vijayawada", "type": "Private"},
    {"name": "Aster Hospital", "location": "Vijayawada", "type": "Private"},
    {"name": "District Hospital", "location": "West Godavari", "type": "Government"},
    {"name": "KIMS Hospital", "location": "Hyderabad", "type": "Private"},
]

# -------------------------------
# Detect intent
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
# Extract disease
# -------------------------------
def extract_disease(query):
    query = query.lower()
    diseases = ["fever", "cold", "cough", "flu", "diabetes"]

    for d in diseases:
        if d in query:
            return d
    return None

# -------------------------------
# Extract only required part
# -------------------------------
def filter_response(text, intent):
    sentences = text.split(".")
    
    result = ""

    for s in sentences:
        s_lower = s.lower()

        if intent == "definition" and ("is a" in s_lower or "is an" in s_lower):
            result += s.strip() + ". "

        elif intent == "symptoms" and (
            "symptom" in s_lower or "chills" in s_lower or "headache" in s_lower
        ):
            result += s.strip() + ". "

        elif intent == "advice" and (
            "advice" in s_lower or "drink" in s_lower or "rest" in s_lower or "take" in s_lower
        ):
            result += s.strip() + ". "

    return result.strip() if result else "No relevant information found."

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("🏥 AI Healthcare Assistant")
st.write("Ask about symptoms, diseases, or health advice")

query = st.text_input("Enter your symptoms:")

# -------------------------------
# Chatbot Response
# -------------------------------
if query:
    query_embedding = model.encode([query])
    D, I = index.search(np.array(query_embedding), k=5)

    intent = detect_intent(query)
    disease = extract_disease(query)

    st.subheader("🤖 AI Response:")

    best_text = None

    for i in I[0]:
        if disease and disease in texts[i].lower():
            best_text = texts[i]
            break

    if not best_text:
        best_text = texts[I[0][0]]

    final_answer = filter_response(best_text, intent)

    st.success(final_answer)

    # 👇 Suggest hospitals
    st.info("💡 Need medical help? Find nearby hospitals below 👇")

# -------------------------------
# Hospital Finder Section
# -------------------------------
st.subheader("🏥 Find Nearby Hospitals")

city = st.text_input("Enter your city:")

if st.button("Show Hospitals"):
    found = False

    for h in hospitals:
        if city and city.lower() in h["location"].lower():
            st.success(f"🏥 {h['name']}")
            st.write(f"📍 Location: {h['location']}")
            st.write(f"🏷️ Type: {h['type']}")
            st.write("---")
            found = True

    if not found:
        st.warning("No hospitals found in this area.")
