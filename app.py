import streamlit as st
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import time

st.set_page_config(page_title="Chat with Web Docs", layout="wide")

st.title("Web-Based Doc Summarization Bot using ChatGroq")

# Sidebar for Configuration
with st.sidebar:
    st.header("Configuration")
    groq_api_key = st.text_input("Enter Groq API Key", type="password")
    
    embedding_provider = st.selectbox(
        "Select Embedding Provider",
        ("Ollama (Local)", "HuggingFace (Cloud/Local)")
    )
    
    if embedding_provider == "Ollama (Local)":
        ollama_model = st.text_input("Ollama Embedding Model", value="nomic-embed-text")
        st.markdown("Ensure Ollama is running locally with the specified model pulled.")
    else:
        st.markdown("Using `all-MiniLM-L6-v2` model from HuggingFace. Runs on CPU, suitable for cloud deployment.")

if not groq_api_key:
    st.info("Please enter your Groq API Key in the sidebar to proceed.")
    st.stop()

# Initialize Session State
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# URL Input
url = st.text_input("Enter Web URL to Summarize/Query")

if st.button("Process URL"):
    if not url:
        st.error("Please enter a URL.")
    else:
        try:
            with st.spinner("Loading and processing content..."):
                # Load Data
                loader = WebBaseLoader(url)
                docs = loader.load()
                
                # Split Data
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                splits = text_splitter.split_documents(docs)
                
                # Create Embeddings
                if embedding_provider == "Ollama (Local)":
                    embeddings = OllamaEmbeddings(model=ollama_model)
                else:
                    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                
                # Create Vector Store
                vector_store = FAISS.from_documents(documents=splits, embedding=embeddings)
                st.session_state.vector_store = vector_store
                
                st.success(f"Successfully processed {len(splits)} chunks from the URL.")
        except Exception as e:
            st.error(f"An error occurred: {e}")

# Chat Interface
if st.session_state.vector_store is not None:
    st.subheader("Ask a Question")
    user_query = st.text_input("What do you want to know about the document?")
    
    if user_query:
        try:
            llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama3-8b-8192")
            
            prompt = ChatPromptTemplate.from_template("""
            Answer the following question based only on the provided context:
            
            <context>
            {context}
            </context>
            
            Question: {input}
            """)
            
            document_chain = create_stuff_documents_chain(llm, prompt)
            retriever = st.session_state.vector_store.as_retriever()
            retrieval_chain = create_retrieval_chain(retriever, document_chain)
            
            with st.spinner("Generating response..."):
                start_time = time.time()
                response = retrieval_chain.invoke({"input": user_query})
                end_time = time.time()
                
                st.write(response["answer"])
                st.caption(f"Response generated in {end_time - start_time:.2f} seconds.")
                
                with st.expander("Document Similarity Search"):
                    for i, doc in enumerate(response["context"]):
                        st.write(f"**Chunk {i+1}:**")
                        st.write(doc.page_content)
                        st.write("---")
        except Exception as e:
            st.error(f"Error generating response: {e}")
