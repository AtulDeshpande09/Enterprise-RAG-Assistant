# streamlit_app.py
import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from pathlib import Path

from langchain_openai import ChatOpenAI
from langchain_community.llms import HuggingFaceHub

load_dotenv()

INDEX_DIR = Path("index")
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True, parents=True)

SYSTEM_PROMPT = """You are a helpful analyst. Answer the user's question using ONLY the provided context.
If the answer is not in the context, say you don't know. Be concise and include a 'Sources:' list with titles or file names.
"""
PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "Question: {question}\n\nContext:\n{context}")
])

@st.cache_resource
def get_embeddings():
    return SentenceTransformerEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

@st.cache_resource
def load_vs(embed):
    return FAISS.load_local(str(INDEX_DIR), embed, allow_dangerous_deserialization=True)

def make_llm():
    provider = os.getenv("LLM_PROVIDER", "openai").lower()
    if provider == "openai":
        return ChatOpenAI(model="gpt-4o-mini", temperature=0)
    else:
        repo_id = os.getenv("HF_REPO_ID", "mistralai/Mistral-7B-Instruct-v0.2")
        return HuggingFaceHub(
            repo_id=repo_id, task="text-generation",
            model_kwargs={"temperature": 0.2, "max_new_tokens": 512}
        )

def format_context(docs):
    lines = []
    for i, d in enumerate(docs, 1):
        meta = d.metadata or {}
        src = meta.get("source", "unknown")
        page = meta.get("page", None)
        tag = f"{src} (p.{page})" if page is not None else src
        lines.append(f"[{i}] {tag}\n{d.page_content[:800]}")
    return "\n\n".join(lines)

def add_files_to_data(uploaded_files):
    for uf in uploaded_files:
        p = DATA_DIR / uf.name
        p.write_bytes(uf.getbuffer())

def rebuild_index(embed):
    docs = []
    for path in DATA_DIR.rglob("*"):
        if path.suffix.lower() == ".pdf":
            docs.extend(PyPDFLoader(str(path)).load())
        elif path.suffix.lower() in {".txt", ".md"}:
            docs.extend(TextLoader(str(path), encoding="utf-8").load())
    if not docs:
        return None
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(docs)
    vs = FAISS.from_documents(chunks, embed)
    vs.save_local(str(INDEX_DIR))
    return vs

def main():
    st.title("Enterprise RAG Assistant")
    st.caption("Upload docs → Build index → Ask questions with citations")

    embed = get_embeddings()

    with st.sidebar:
        st.header("Index Builder")
        files = st.file_uploader("Upload PDFs / .txt / .md", type=["pdf", "txt", "md"], accept_multiple_files=True)
        if st.button("Add to data folder"):
            if files:
                add_files_to_data(files)
                st.success("Files saved to ./data")
            else:
                st.warning("No files selected.")

        if st.button("Rebuild index"):
            vs = rebuild_index(embed)
            if vs:
                st.success("Index rebuilt.")
            else:
                st.warning("No documents found in ./data")

    if not INDEX_DIR.exists():
        st.info("No index found. Upload files in the sidebar and click 'Rebuild index'.")
        return

    try:
        vs = load_vs(embed)
    except Exception:
        st.warning("Index missing or incompatible. Rebuild in sidebar.")
        return

    llm = make_llm()
    chain = PROMPT | llm | StrOutputParser()

    question = st.text_input("Ask a question about your documents:")
    k = st.slider("Top-k chunks", 2, 8, 4)

    if st.button("Ask") and question.strip():
        docs = vs.as_retriever(search_type="similarity", search_kwargs={"k": k}).get_relevant_documents(question)
        if not docs:
            st.write("No relevant context found.")
            return
        context = format_context(docs)
        answer = chain.invoke({"question": question, "context": context})
        st.markdown("### Answer")
        st.write(answer)
        st.markdown("### Sources")
        for i, d in enumerate(docs, 1):
            meta = d.metadata or {}
            src = meta.get("source", "unknown")
            page = meta.get("page", None)
            st.write(f"[{i}] {src}" + (f" (p.{page})" if page is not None else ""))

if __name__ == "__main__":
    main()

