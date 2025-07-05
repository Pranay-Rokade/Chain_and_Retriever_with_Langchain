# 📚 LangChain RAG QA Pipeline – Chains and Retrieval with PDF Data

This notebook demonstrates how to build a **retrieval-augmented question-answering (RAG)** pipeline using [LangChain](https://www.langchain.com/). Specifically, it covers:

- ✅ PDF document ingestion and chunking
- ✅ Embedding generation with HuggingFace
- ✅ Vector storage and retrieval using FAISS
- ✅ Designing prompts for retrieval-based QA
- ✅ Building document chains using:
  - `create_stuff_documents_chain`
  - `create_retrieval_chain`
- ✅ Querying the chain to get natural language answers from context

---

## 🧠 What You’ll Learn

| Concept | Description |
|--------|-------------|
| `create_stuff_documents_chain` | Wraps an LLM with a prompt that receives all retrieved docs as context |
| `create_retrieval_chain` | Combines a retriever with a document chain for end-to-end querying |
| Prompt Engineering | Custom, role-driven prompt to guide the LLM output |
| FAISS Vector Search | Perform semantic similarity-based document retrieval |
| HuggingFace Embeddings | Use pretrained sentence transformers for dense vector representation |

---

## 🛠️ Pipeline Summary

### 🔹 Step 1: PDF Data Ingestion

```python
from langchain_community.document_loaders import PyPDFLoader
````

### 🔹 Step 2: Chunking the Text

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter
```

### 🔹 Step 3: Embedding & Vector Store

```python
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
```

### 🔹 Step 4: Retriever + Query

```python
retriever = db.as_retriever()
result = db.similarity_search(query)
```

---

## 🧠 QA Chain Setup

### 1️⃣ Load LLM with Ollama

```python
from langchain_community.llms import Ollama
```

### 2️⃣ Prompt Design

```python
from langchain_core.prompts import ChatPromptTemplate
```

### 3️⃣ Chain Creation

```python
from langchain.chains.combine_documents import create_stuff_documents_chain
document_chain = create_stuff_documents_chain(llm, prompt)

from langchain.chains import create_retrieval_chain
retrieval_chain = create_retrieval_chain(retriever, document_chain)
```

---

## 🧪 Querying the RAG Chain

```python
response = retrieval_chain.invoke({})
```

---

## ✅ Key Takeaways

* You now understand how to integrate **retrievers**, **chains**, and **custom prompts** into a functional RAG pipeline.
* You explored the **`create_stuff_documents_chain`** to pass retrieved docs directly into the prompt.
* Used **local Ollama** LLMs for offline, privacy-preserving inference.

---

## 🔮 Future Exploration Ideas

* Use `ConversationalRetrievalChain` for multi-turn context
* Replace FAISS with `Chroma` or `Weaviate`
* Serve as an API using `LangServe` or `FastAPI`
* Enable LangSmith logging to debug and analyze chains

---

## 🧠 Credits

* [LangChain Documentation](https://docs.langchain.com/)
* [Ollama](https://ollama.ai/)
* [HuggingFace Sentence Transformers](https://www.sbert.net/)
* [FAISS by Meta AI](https://github.com/facebookresearch/faiss)

---

## 🚀 Ready to Build Smarter QA Systems with LangChain!
