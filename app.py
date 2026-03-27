import streamlit as st
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Load model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load medical data
with open("data/medical.txt", "r") as f:
    texts = f.readlines()

# Create embeddings
embeddings = model.encode(texts)

# Create FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

# Streamlit UI
st.title("🏥 AI Healthcare Assistant")

query = st.text_input("Enter your symptoms:")

if query:
    query_embedding = model.encode([query])
    D, I = index.search(np.array(query_embedding), k=1)
    result = texts[I[0][0]]

    st.write("🤖", result)
