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
# ✅ FIXED: Convert city → coordinates
# -------------------------------
def get_coordinates(city):
    url = "https://nominatim.openstreetmap.org/search"
    headers = {"User-Agent": "my-app"}

    params = {
        "q": city,
        "format": "json"
    }

    try:
        res = requests.get(url, params=params, headers=headers, timeout=10)
        data = res.json()

        if data:
            return float(data[0]["lat"]), float(data[0]["lon"])
    except:
        pass

    return None, None


# -------------------------------
# ✅ FIXED: Get hospitals (stable)
# -------------------------------
def get_hospitals(city):
    lat, lon = get_coordinates(city)

    if not lat:
        return []

    url = "https://overpass-api.de/api/interpreter"
    headers = {"User-Agent": "my-app"}

    query = f"""
    [out:json][timeout:25];
    node["amenity"="hospital"](around:15000,{lat},{lon});
    out;
    """

    try:
        res = requests.post(url, data=query, headers=headers, timeout=20)
        data = res.json()

        hospitals = []

        for el in data.get("elements", []):
            name = el.get("tags", {}).get("name", "Unknown Hospital")

            hospitals.append({
                "name": name,
                "lat": el.get("lat"),
                "lon": el.get("lon")
            })

        return hospitals

    except:
        return []


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
# Hospital Finder
# -------------------------------
st.subheader("🏥 Nearby Hospitals")

city = st.text_input("Enter city (try Vijayawada / Hyderabad):")

if st.button("Search Hospitals"):
    if city:
        with st.spinner("Loading..."):
            hospitals = get_hospitals(city)

        if hospitals:
            st.success(f"{len(hospitals)} hospitals found")

            for h in hospitals[:5]:
                st.write("🏥", h["name"])

            df = pd.DataFrame(hospitals)
            st.map(df)

        else:
            st.error("No hospitals found (try Vijayawada)")
    else:
        st.warning("Enter city")
