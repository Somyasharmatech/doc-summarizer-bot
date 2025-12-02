# Web-Based Doc Summarization Bot

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://doc-summarizer-bot.streamlit.app/)

A conversational AI bot built with Streamlit that allows users to chat with and summarize web-based documents. It leverages LangChain, Groq (Llama 3), and Ollama for efficient document processing and retrieval.

## 🚀 Live Demo
**[Click here to use the App](https://doc-summarizer-bot.streamlit.app/)**
*(Note: If the link doesn't work, check your Streamlit Cloud dashboard for the exact URL)*


## Features

-   **Web Document Loading**: Fetches text content from any provided URL.
-   **Intelligent Chunking**: Splits large documents into manageable chunks for processing.
-   **Vector Search**: Uses FAISS and Ollama embeddings for fast similarity search.
-   **Conversational Interface**: Chat with the document using Groq's Llama 3 model.
-   **Source Citations**: Shows the exact chunks used to generate the answer.

## Tech Stack

-   **Frontend**: Streamlit
-   **LLM**: Groq (Llama 3)
-   **Embeddings**: Ollama (`nomic-embed-text` or similar)
-   **Vector Store**: FAISS
-   **Orchestration**: LangChain

## Prerequisites

1.  **Python 3.8+**
2.  **Groq API Key**: Get one from [console.groq.com](https://console.groq.com/).
3.  **Ollama**: Installed and running locally.
    -   Download from [ollama.com](https://ollama.com/).
    -   Pull an embedding model: `ollama pull nomic-embed-text` (or `llama3`, etc.).

## Installation

1.  Clone the repository:
    ```bash
    git clone <repository-url>
    cd doc_summarizer_bot
    ```

2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1.  Start the Streamlit app:
    ```bash
    streamlit run app.py
    ```

2.  In the sidebar:
    -   Enter your **Groq API Key**.
    -   Specify the **Ollama Embedding Model** (default: `nomic-embed-text`).

3.  Enter a URL in the main input field and click **Process URL**.

4.  Ask questions about the content in the chat interface!

## License

MIT
