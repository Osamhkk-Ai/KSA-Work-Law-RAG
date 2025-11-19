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

```
project/
│
├── app.py
├── backend_api.py
├── build_chroma.py
├── query_chroma.py
├── index.html
│
├── data/
├── frontend/
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

## ⚠️ Security Notes

Before publishing this repo, ensure you **do NOT commit**:

- pass.env  
- Any API keys  
- chroma_db/  
- __pycache__/  
- Temporary experiments/notebooks  

---

## 🛠 How to Run Locally

### 1️⃣ Install dependencies:
```
pip install -r requirements.txt
```

### 2️⃣ Build the vector database:
```
python build_chroma.py
```

### 3️⃣ Run the backend API:
```
python backend_api.py
```

### 4️⃣ Open the frontend:
Open index.html in your browser.

---

## 📥 Data Source  
Manually structured dataset from the Saudi Labor Law including:

- Article number  
- Part  
- Chapter  
- Title  
- Article text  
- Explanation  

---

## 📘 Example Output

**Question:**  
> ما هي حقوقي عند انتهاء العقد غير محدد المدة؟

**Answer:**  
- النظام ينص على …  
- المادة المستخدمة: *المادة ٧٧*  
- يعرض النظام النص الكامل للمادة أيضًا.

---

## 🎯 Purpose  
Built for:

- HR teams  
- Legal assistants  
- Employees & employers  
- Educational & portfolio use  

---

## 🤝 Contributing  
Pull requests are welcome.

---

## 📩 Contact  
**Developer:** Osama Alghamdi

---

## ⭐ Star the Repo  
إذا أعجبك المشروع لا تنسَ تعمل Star 🌟
