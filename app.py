import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

# -------------------------------
# Load Documents
# -------------------------------
loader = TextLoader("data/medical.txt")
documents = loader.load()

# -------------------------------
# Split Text
# -------------------------------
text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=20)
docs = text_splitter.split_documents(documents)

# -------------------------------
# Embeddings
# -------------------------------
embeddings = HuggingFaceEmbeddings()

# -------------------------------
# Vector DB (FAISS)
# -------------------------------
db = FAISS.from_documents(docs, embeddings)

# -------------------------------
# QA Chain
# -------------------------------
qa = RetrievalQA.from_chain_type(
    llm=OpenAI(temperature=0),
    chain_type="stuff",
    retriever=db.as_retriever()
)

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("🏥 AI Healthcare Assistant")

query = st.text_input("Enter your symptoms or question:")

if query:
    response = qa.run(query)
    st.write("🤖", response)
