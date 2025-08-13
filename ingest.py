# ingest.py
import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

DATA_DIR = Path("data")
INDEX_DIR = Path("index")
INDEX_DIR.mkdir(exist_ok=True, parents=True)

def load_docs():
    docs = []
    for path in DATA_DIR.rglob("*"):
        if path.suffix.lower() == ".pdf":
            try:
                docs.extend(PyPDFLoader(str(path)).load())
            except Exception as e:
                print(f"[warn] PDF load failed {path}: {e}")
        elif path.suffix.lower() in {".txt", ".md"}:
            try:
                docs.extend(TextLoader(str(path), encoding="utf-8").load())
            except Exception as e:
                print(f"[warn] Text load failed {path}: {e}")
    return docs

def main():
    if not DATA_DIR.exists():
        raise SystemExit("Put your source files in ./data first.")

    print("Loading documents...")
    docs = load_docs()
    if not docs:
        raise SystemExit("No documents found in ./data. Add PDFs or .txt/.md and re-run.")

    print(f"Loaded {len(docs)} documents. Splitting...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        add_start_index=True,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    print(f"Created {len(chunks)} chunks.")

    print("Embedding...")
    # Great default multilingual embedding model; works well for most corpora
    embed = SentenceTransformerEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    print("Building FAISS index...")
    vs = FAISS.from_documents(chunks, embed)
    vs.save_local(str(INDEX_DIR))
    print("Index saved to ./index")

if __name__ == "__main__":
    main()

