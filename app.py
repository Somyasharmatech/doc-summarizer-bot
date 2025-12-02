import streamlit as st
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
import time

st.set_page_config(page_title="Chat with Web Docs", layout="wide")

st.title("Web-Based Doc Summarization Bot using ChatGroq")

# Sidebar for Configuration
with st.sidebar:
    st.header("Configuration")
    
    # Secrets Management
    if "GROQ_API_KEY" in st.secrets:
        groq_api_key = st.secrets["GROQ_API_KEY"]
        st.success("Groq API Key loaded from secrets!")
    else:
        groq_api_key = st.text_input("Enter Groq API Key", type="password")

    # Model Selector
    model_option = st.selectbox(
        "Select Model",
        ("llama-3.3-70b-versatile", "mixtral-8x7b-32768", "gemma-7b-it")
    )
    
    embedding_provider = st.selectbox(
        "Select Embedding Provider",
        ("Ollama (Local)", "HuggingFace (Cloud/Local)")
    )
    
    if embedding_provider == "Ollama (Local)":
        ollama_model = st.text_input("Ollama Embedding Model", value="nomic-embed-text")
        st.markdown("Ensure Ollama is running locally with the specified model pulled.")
    else:
        st.markdown("Using `all-MiniLM-L6-v2` model from HuggingFace. Runs on CPU, suitable for cloud deployment.")

    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

if not groq_api_key:
    st.info("Please enter your Groq API Key in the sidebar to proceed.")
    st.stop()

# Initialize Session State
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# URL Input
url = st.text_input("Enter Web URL to Summarize/Query")

col1, col2 = st.columns([1, 1])

with col1:
    process_button = st.button("Process URL")

with col2:
    summarize_button = st.button("Summarize Document")

if process_button:
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

# Summarization Logic
if summarize_button:
    if st.session_state.vector_store is None:
        st.error("Please process a URL first.")
    else:
        try:
            llm = ChatGroq(groq_api_key=groq_api_key, model_name=model_option)
            
            # We use the vector store to retrieve relevant content for summarization
            # Ideally we would pass the whole document, but for RAG we simulate it by asking for a summary
            
            prompt = ChatPromptTemplate.from_template("""
            Summarize the following content in a concise and structured manner:
            
            <context>
            {context}
            </context>
            """)
            
            document_chain = create_stuff_documents_chain(llm, prompt)
            retriever = st.session_state.vector_store.as_retriever()
            retrieval_chain = create_retrieval_chain(retriever, document_chain)
            
            with st.spinner("Generating summary..."):
                response = retrieval_chain.invoke({"input": "Summarize the main points of this document."})
                st.markdown("### Document Summary")
                st.write(response["answer"])
                
        except Exception as e:
            st.error(f"Error generating summary: {e}")

# Chat Interface
if st.session_state.vector_store is not None:
    st.subheader("Chat with Document")
    
    # Display Chat History
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if user_query := st.chat_input("Ask a question..."):
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)
            
        try:
            llm = ChatGroq(groq_api_key=groq_api_key, model_name=model_option)
            
            # History Aware Retriever
            contextualize_q_system_prompt = """Given a chat history and the latest user question \
            which might reference context in the chat history, formulate a standalone question \
            which can be understood without the chat history. Do NOT answer the question, \
            just reformulate it if needed and otherwise return it as is."""
            
            contextualize_q_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", contextualize_q_system_prompt),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}"),
                ]
            )
            
            history_aware_retriever = create_history_aware_retriever(
                llm, st.session_state.vector_store.as_retriever(), contextualize_q_prompt
            )
            
            # Answer Generation
            qa_system_prompt = """You are an assistant for question-answering tasks. \
            Use the following pieces of retrieved context to answer the question. \
            If you don't know the answer, just say that you don't know. \
            Use three sentences maximum and keep the answer concise.\

            {context}"""
            
            qa_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", qa_system_prompt),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}"),
                ]
            )
            
            question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
            rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
            
            # Convert session history to LangChain format
            chat_history = []
            for msg in st.session_state.messages:
                if msg["role"] == "user":
                    chat_history.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    chat_history.append(AIMessage(content=msg["content"]))
            
            with st.spinner("Thinking..."):
                response = rag_chain.invoke({"input": user_query, "chat_history": chat_history})
                answer = response["answer"]
                
                st.session_state.messages.append({"role": "assistant", "content": answer})
                with st.chat_message("assistant"):
                    st.markdown(answer)
                    
                with st.expander("Source Documents"):
                    for i, doc in enumerate(response["context"]):
                        st.markdown(f"**Chunk {i+1}**")
                        st.text(doc.page_content)
                        
        except Exception as e:
            st.error(f"Error generating response: {e}")
