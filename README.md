# KSA-Work-Law-RAG
# 🇸🇦 Saudi Labor Law RAG  
**An intelligent Retrieval-Augmented Generation (RAG) system for answering questions related to the Saudi Labor Law (نظام العمل السعودي).**

This project provides accurate, citation-based answers to user questions by searching, retrieving, and reasoning over official Saudi Labor Law articles.  
It supports question rewriting, hybrid search, and detailed legal references.

---

## 🚀 Features

### 🔍 1. Smart Question Rewriting  
The system rewrites user queries to clearer, law-friendly formats to improve retrieval accuracy.

### 📚 2. Hybrid Search (BM25 + Embeddings)  
Combines:
- **Semantic search** using Jina Embeddings  
- **Keyword/BM25 search**  
for maximum precision.

### 🧠 3. Context-Aware Answer Generation  
Uses a language model to:
- Understand the rewritten question  
- Retrieve the most relevant legal articles  
- Generate a clear, structured answer  
- Provide **full article text** for transparency

### 📑 4. ChromaDB Vector Store  
All articles are embedded and stored locally using Chroma.

---

## 🗂 Project Structure

