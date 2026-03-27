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
# Extract disease from query
# -------------------------------
def extract_disease(query):
    query = query.lower()
    diseases = ["fever", "cold", "cough", "flu", "diabetes"]

    for d in diseases:
        if d in query:
            return d
    return None

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("🏥 AI Healthcare Assistant")
st.write("Ask about symptoms, diseases, or health advice")

query = st.text_input("Enter your symptoms:")

# -------------------------------
# Search (Fixed)
# -------------------------------
if query:
    query_embedding = model.encode([query])

    # 🔥 Get more results for filtering
    D, I = index.search(np.array(query_embedding), k=5)

    disease = extract_disease(query)

    st.subheader("🤖 AI Response:")

    best_result = None

    # ✅ Filter only correct disease
    for i in I[0]:
        if disease and disease in texts[i].lower():
            best_result = texts[i]
            break

    # ✅ fallback (if nothing matches)
    if not best_result:
        best_result = texts[I[0][0]]

    # ✅ remove duplicates (clean output)
    best_result = best_result.strip()

    st.write("👉", best_result)
