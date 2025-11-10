**Enterprise-RAG-Assistant**

A general-purpose Retrieval Augmented Generation (RAG) pipeline that ingests PDF, HTML, or plain text documents, indexes them, and answers user queries with source citations. This project is intended for internal knowledge search, documentation Q&A, and enterprise data assistance.

### Features

* Upload and process unstructured documents (PDF/HTML/Text)
* Text chunking and embedding-based retrieval
* Query answering with cited references
* Local vector database for efficient semantic search
* Streamlit UI for interactive use

### Tech Stack

* **Core Libraries**:
  langchain
  langchain-community
  langchain-text-splitters
  sentence-transformers
  pypdf
  python-dotenv

* **Vector Store**:
  faiss-cpu

* **UI**:
  streamlit

* **Model Hub**:
  huggingface_hub

* **LLM Provider**:
  OpenAI API

### Setup

1. Clone the repository
2. Install dependencies:

   ```
   pip install -r requirements.txt
   ```
3. Add API keys to `.env`
4. Run the application:

   ```
   streamlit run app.py
   ```

### Usage

Upload documents, allow embeddings to generate, and query the knowledge base. The system retrieves relevant text chunks and produces responses with citations.
