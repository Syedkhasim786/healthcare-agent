import streamlit as st
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# -------------------------------
# Load model
# -------------------------------
model = SentenceTransformer("all-MiniLM-L6-v2")

# -------------------------------
# Load medical data (FIXED)
# -------------------------------
with open("data/medical.txt", "r") as f:
    text = f.read()

# Split into meaningful sections (IMPORTANT FIX)
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
# Streamlit UI
# -------------------------------
st.title("🏥 AI Healthcare Assistant")
st.write("Ask about symptoms, diseases, or health advice")

query = st.text_input("Enter your symptoms:")

# -------------------------------
# Search (Improved)
# -------------------------------
if query:
    query_embedding = model.encode([query])
    
    # Get top 2 results instead of 1 (better answers)
    D, I = index.search(np.array(query_embedding), k=2)

    st.subheader("🤖 AI Response:")
    
    for i in I[0]:
        st.write("👉", texts[i])
