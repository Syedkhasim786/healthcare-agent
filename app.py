import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import OpenAI
from langchain.chains.retrieval_qa.base import RetrievalQA

# Load documents
loader = TextLoader("data/medical.txt")
documents = loader.load()

# Split text
text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=20)
docs = text_splitter.split_documents(documents)

# Embeddings
embeddings = HuggingFaceEmbeddings()

# Vector DB
db = FAISS.from_documents(docs, embeddings)

# QA Chain
qa = RetrievalQA.from_chain_type(
    llm=OpenAI(temperature=0),
    retriever=db.as_retriever()
)

# UI
st.title("🏥 AI Healthcare Assistant")

query = st.text_input("Enter your symptoms or question:")

if query:
    response = qa.run(query)
    st.write("🤖", response)
