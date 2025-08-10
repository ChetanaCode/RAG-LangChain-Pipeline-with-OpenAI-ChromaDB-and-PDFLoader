"""
README with Code
================
Code modified.

This file contains both the README and the full main.py script with type hints.

Steps to Run:
1. Ensure you have `.env` with `OPENAI_API_KEY` and `LANGSMITH_API_KEY` set.
2. Install dependencies:
   pip install python-dotenv langchain langchain_openai langchain_chroma langchain_community langsmith pypdf
3. Place your PDF file in the same directory as this script.
4. Run the script:
   python README_with_code.py
"""

from dotenv import load_dotenv

load_dotenv()
import os
from langchain.chat_models import init_chat_model
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langsmith import Client
from langchain.schema import Document
from typing import List

# Create OpenAI Chat Model
model = init_chat_model(
    "gpt-4o-mini",
    model_provider="openai",
    api_key=os.environ.get("OPENAI_API_KEY")
)

# Loading PDF document
loader: PyPDFLoader = PyPDFLoader("EPGPMachineLearningAIBrochure__1688114020619.pdf")
docs: List[Document] = loader.load()

# Text splitter for PDF document
text_splitter: RecursiveCharacterTextSplitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits: List[Document] = text_splitter.split_documents(docs)

# OpenAI Embeddings
emb: OpenAIEmbeddings = OpenAIEmbeddings(model="text-embedding-3-small")
texts: List[str] = [doc.page_content for doc in all_splits]

# Create Chroma store
vector_store: Chroma = Chroma(
    collection_name="my_collection",
    embedding_function=emb,
    persist_directory="chroma_db"
)

# Add texts to vector store
vector_store.add_texts(texts=texts, embeddings=emb)

# RAG Prompt
client: Client = Client(api_key=os.environ["LANGSMITH_API_KEY"])
prompt = client.pull_prompt("rlm/rag-prompt", include_model=True)

retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

def format_docs(docs: List[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = ({
                "context": retriever | format_docs,
                "question": RunnablePassthrough()
            }
            | prompt
            | model
            | StrOutputParser()
)

if __name__ == "__main__":
    query: str = "How many years experience does chetana have?"
    answer: str = rag_chain.invoke(query)
    print("Q:", query)
    print("A:", answer)
