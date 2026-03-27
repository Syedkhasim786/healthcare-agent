import streamlit as st
import faiss
import numpy as np
import pandas as pd
import requests
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
# 🔥 REAL Hospital API (OpenStreetMap)
# -------------------------------
def get_hospitals(city):
    try:
        url = "https://overpass-api.de/api/interpreter"

        query = f"""
        [out:json];
        area["name"="{city}"]->.searchArea;
        (
          node["amenity"="hospital"](area.searchArea);
          way["amenity"="hospital"](area.searchArea);
          relation["amenity"="hospital"](area.searchArea);
        );
        out center;
        """

        response = requests.get(url, params={'data': query}, timeout=10)
        data = response.json()

        hospitals = []

        for element in data.get('elements', []):
            name = element.get('tags', {}).get('name', 'Unknown Hospital')

            lat = element.get('lat') or element.get('center', {}).get('lat')
            lon = element.get('lon') or element.get('center', {}).get('lon')

            if lat and lon:
                hospitals.append({
                    "name": name,
                    "lat": lat,
                    "lon": lon
                })

        return hospitals

    except:
        return []

# -------------------------------
# Detect intent
# -------------------------------
def detect_intent(query):
    query = query.lower()
    if any(word in query for word in ["advice", "treatment", "cure", "remedy"]):
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

    st.info("💡 Need medical help? Find real nearby hospitals below 👇")

# -------------------------------
# 🏥 REAL Hospital Finder
# -------------------------------
st.subheader("🏥 Find Real Nearby Hospitals")

city = st.text_input("Enter your city:")

if st.button("Show Real Hospitals"):
    if city:
        with st.spinner("Fetching hospitals..."):
            hospitals = get_hospitals(city)

        if hospitals:
            st.success(f"Found {len(hospitals)} hospitals")

            # Show top 5 names
            for h in hospitals[:5]:
                st.write(f"🏥 {h['name']}")

            # Show map
            df = pd.DataFrame(hospitals)
            st.map(df[['lat', 'lon']])

        else:
            st.warning("No hospitals found or API error.")
    else:
        st.warning("Please enter a city.")
