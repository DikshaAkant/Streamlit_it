import os
import streamlit as st
from dotenv import load_dotenv

# Langchain imports 
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS #find keyword create chain

#Window+.
# Step1 : Page Configuration 
st.set_page_config(page_title="C++ Rag Chatbot", page_icon="💬")
st.title("💬 C++ Rag Chatbot")
st.write("Ask any question related to C++ introduction ")

#Step 2: Load Environment Variables
load_dotenv()

# Step 3: Cache document loading
@st.cache_resource()
def load_vector_store():
    # Step A: Load Documents
    loader = TextLoader("C++_Introduction.txt", encoding = "utf-8")
    documents = loader.load()
    # Step B: Split text
    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 200,
    chunk_overlap = 20
    # Chunk Overlap - 20 Characters Overlap
    # Overlap helps maintain context continuity 
    )

    final_documents = text_splitter.split_documents(documents)
    
    # Step C: Embeddings
    embedding = HuggingFaceEmbeddings(
        model_name = "all-miniLM-L6-v2"
        # This is teh embedding model 
    ) 
    # Step D: Create FAISS Vector Store
    # Converts each chunk to embedding , then stores them and makes search 
    def create_faiss_db(final_documents, embeddings):
         db = FAISS.from_documents(final_documents,embeddings)
         #return faiss database
         return db

#Vector database runs only once because of cache concepts
db = load_vector_store()

#User Input 
query = st.text_input("Enter your question about c++")

if(query):
    # Converts user questions to embeddings 
    # search FAISS database
    # Returns top3 similar chunks 
    document = db.similarity_search(query, k=3)

    st.subheader("📒 Retrieve context")

    for i,doc in enumerate(document):
        st.markdown(f"**Result {i+1} : **")
        st.write(doc.page_content)
