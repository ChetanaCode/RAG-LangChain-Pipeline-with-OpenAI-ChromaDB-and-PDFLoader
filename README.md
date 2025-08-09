# 📄 RAG Pipeline with Chroma, LangChain, and OpenAI

This project demonstrates how to build a **Retrieval-Augmented Generation (RAG)** pipeline using:

- **LangChain** for orchestration
- **Chroma** as the vector store
- **OpenAI** for embeddings and chat completion
- **PyPDFLoader** for document ingestion
- **LangSmith** for prompt management

---

## 🚀 Features

- 📥 **PDF Ingestion** — Load and split large PDFs into manageable chunks.
- 🧠 **Vector Search** — Store and retrieve document embeddings with Chroma.
- 💬 **OpenAI Chat Model** — Use `gpt-4o-mini` for answering questions.
- 🔍 **Retriever Pipeline** — Query your document collection efficiently.
- 🛠 **Prompt Management** — Pull and manage prompts from LangSmith.

---

## 📦 Prerequisites

1. **Python** 3.10+
2. An **OpenAI API key**  
3. A **LangSmith API key**
4. Git & pip installed

---

## 📂 Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Mac/Linux
.venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## 🔑 Environment Variables

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_openai_api_key_here
LANGSMITH_API_KEY=your_langsmith_api_key_here
```

---

## 📜 Usage

### 1. Place Your PDF
Put your PDF in the project root (or specify the path in the script).

### 2. Run the Pipeline

```bash
python main.py
```

### 3. Example Query

```python
rag_chain.invoke("what is the duration of the course?")
```

**Output Example:**
```
The duration of the course is 13 months. It is delivered in a blended format, combining asynchronous learning and live sessions. Students are expected to commit approximately 15 hours per week.
```

---

## 🛠 Project Structure

```
📦 rag-pipeline
├── main.py                # Main pipeline script
├── requirements.txt       # Python dependencies
├── .env                   # API keys
├── chroma_db/             # Persisted vector store
├── EPGPMachineLearningAIBrochure__1688114020619.pdf  # Sample PDF
└── README.md              # This file
```

---

## 📚 How It Works

1. **Load PDF** → `PyPDFLoader` reads and splits the document.
2. **Chunking** → `RecursiveCharacterTextSplitter` creates overlapping text chunks.
3. **Embedding** → `OpenAIEmbeddings` generates vector representations.
4. **Vector Store** → `Chroma` stores embeddings in a local directory.
5. **Retriever** → Retrieves top-k most relevant chunks.
6. **Prompt Chain** → Combines retrieved context with the question for the model.
7. **Model Output** → Returns an answer using GPT-4o-mini.

---

## 🧩 Dependencies

```
langchain
langchain-community
langchain-chroma
langchain-openai
pypdf
python-dotenv
```

Install them with:
```bash
pip install -r requirements.txt
```

---

## 📜 License
MIT License — feel free to use and modify for personal or commercial projects.
