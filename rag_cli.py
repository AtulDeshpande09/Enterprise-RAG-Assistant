# rag_cli.py
import os
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.docstore.document import Document
from langchain.schema.output_parser import StrOutputParser

# LLM providers
from langchain_openai import ChatOpenAI  # needs openai>=1.0 and langchain_openai
from langchain_community.llms import HuggingFaceHub

load_dotenv()

INDEX_DIR = "index"

SYSTEM_PROMPT = """You are a helpful analyst. Answer the user's question using ONLY the provided context.
If the answer is not in the context, say you don't know. Be concise and include a 'Sources:' list with titles or file names.
"""

PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "Question: {question}\n\nContext:\n{context}")
])

def load_vectorstore():
    embed = SentenceTransformerEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    return FAISS.load_local(INDEX_DIR, embed, allow_dangerous_deserialization=True)

def make_llm():
    provider = os.getenv("LLM_PROVIDER", "openai").lower()
    if provider == "openai":
        # cheap & solid default if you have an API key
        return ChatOpenAI(model="gpt-4o-mini", temperature=0)
    elif provider == "hf":
        repo_id = os.getenv("HF_REPO_ID", "mistralai/Mistral-7B-Instruct-v0.2")
        return HuggingFaceHub(
            repo_id=repo_id,
            task="text-generation",
            model_kwargs={"temperature": 0.2, "max_new_tokens": 512}
        )
    else:
        raise ValueError("LLM_PROVIDER must be 'openai' or 'hf'")

def format_context(docs):
    lines = []
    for i, d in enumerate(docs, 1):
        meta = d.metadata or {}
        src = meta.get("source", "unknown")
        page = meta.get("page", None)
        tag = f"{src} (p.{page})" if page is not None else src
        lines.append(f"[{i}] {tag}\n{d.page_content[:800]}")
    return "\n\n".join(lines)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("question", nargs="+", help="Your question")
    parser.add_argument("--k", type=int, default=4, help="Top-k chunks to retrieve")
    args = parser.parse_args()

    question = " ".join(args.question)

    vs = load_vectorstore()
    retriever = vs.as_retriever(search_type="similarity", search_kwargs={"k": args.k})
    docs = retriever.get_relevant_documents(question)

    if not docs:
        print("No relevant context found.")
        return

    llm = make_llm()
    chain = PROMPT | llm | StrOutputParser()

    context = format_context(docs)
    answer = chain.invoke({"question": question, "context": context})
    print("\n=== Answer ===\n")
    print(answer)
    print("\n=== Retrieved Sources ===")
    for i, d in enumerate(docs, 1):
        meta = d.metadata or {}
        src = meta.get("source", "unknown")
        page = meta.get("page", None)
        print(f"[{i}] {src}" + (f" (p.{page})" if page is not None else ""))

if __name__ == "__main__":
    main()

