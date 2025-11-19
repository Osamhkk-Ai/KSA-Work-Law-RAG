from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
from dotenv import load_dotenv
import os
import asyncio

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder
import google.generativeai as genai

# ========== تحميل المفاتيح ==========
load_dotenv()
load_dotenv("pass.env")

genai.configure(api_key=os.getenv("google_api_key"))

# ========== إعداد FastAPI ==========
app = FastAPI(title="Legal RAG API", version="1.0.0")

# إضافة CORS للسماح بالاتصال من المتصفح
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========== Models للطلبات والردود ==========
class QueryRequest(BaseModel):
    query: str

class ArticleResponse(BaseModel):
    title: str
    article_id: str
    content: str
    part: Optional[str] = ""
    chapter: Optional[str] = ""

class AnswerResponse(BaseModel):
    question: str
    legal_answer: str
    referenced_article: str
    simplified: str

class SearchResponse(BaseModel):
    rewritten_query: str
    articles: List[ArticleResponse]
    answer: AnswerResponse
    search_type: str

# ========== تهيئة النماذج ==========
print("🔄 جاري تحميل النماذج...")

rewriter_llm = genai.GenerativeModel("gemini-2.5-flash-lite")
answer_llm = genai.GenerativeModel("gemini-2.5-flash")

embeddings = HuggingFaceEmbeddings(
    model_name="jinaai/jina-embeddings-v3",
    model_kwargs={"trust_remote_code": True}
)

vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

reranker = CrossEncoder(
    "Alibaba-NLP/gte-multilingual-reranker-base",
    trust_remote_code=True
)

# تحميل البيانات
df = pd.read_csv(
    "./data/nzam-3ml-combined-text-final.csv",
    dtype=str
)

print("✅ تم تحميل جميع النماذج بنجاح!")

# ========== دوال مساعدة ==========
async def rewrite_query(query: str) -> str:
    """إعادة صياغة السؤال"""
    rewriter_prompt = f"""
You are an Arabic question rewriter assistant.
Rules:
- Rewrite the user's question into a clear and precise legal query suitable for semantic search (RAG).
- If the user clearly asks for an article number, return ONLY the number like: 25 or 1 or 123 
- If the question has a number but NOT referring to an article, ignore it
- Fix spelling mistakes silently
- Output Arabic only

السؤال:
{query}
"""
    
    rewritten_obj = await asyncio.to_thread(
        rewriter_llm.generate_content, 
        rewriter_prompt
    )
    return rewritten_obj.text.strip()

async def search_direct_article(article_id: str) -> List[dict]:
    """البحث المباشر عن مادة محددة"""
    match = df[df["article_id"] == article_id]
    
    results = []
    if not match.empty:
        for _, row in match.iterrows():
            results.append({
                "title": row["title"],
                "article_id": row["article_id"],
                "content": f"(المادة {row['article_id']})\n{row['text']}",
                "part": row.get("part", ""),
                "chapter": row.get("chapter", "")
            })
    
    return results

async def search_semantic(query: str, k: int = 12, top_n: int = 5) -> List[dict]:
    """البحث الدلالي مع Re-ranking"""
    # البحث الأولي
    docs = await asyncio.to_thread(
        vectorstore.similarity_search,
        query,
        k=k
    )
    
    # Re-ranking
    pairs = [[query, doc.page_content] for doc in docs]
    scores = await asyncio.to_thread(reranker.predict, pairs)
    
    scored_docs = list(zip(docs, scores))
    scored_docs = sorted(scored_docs, key=lambda x: x[1], reverse=True)
    
    # أفضل النتائج
    top_docs = [doc for doc, score in scored_docs[:top_n]]
    
    results = []
    for doc in top_docs:
        meta = doc.metadata
        results.append({
            "title": meta.get("title", ""),
            "article_id": str(meta.get("article_id", "")), 
            "content": f"(المادة {meta.get('article_id')})\n{doc.page_content}",
            "part": meta.get("part", ""),
            "chapter": meta.get("chapter", "")
        })

    
    return results

async def generate_answer(query: str, articles: List[dict]) -> dict:
    """توليد الإجابة النهائية"""
    # تجهيز السياق
    context = "\n\n".join([art["content"] for art in articles])
    
    final_prompt = f"""
You are an advanced Arabic legal assistant.

Tasks:
1-Correct the user's question without changing its meaning.
2-Use only the official legal texts when generating the answer.
3-If no direct legal text exists, respond with: "There is no direct article in the law for this case."
4-Answer in Arabic only.

---

📚 **Legal Context:**
{context}

❓ **Original User Question:**
{query}

**Write in the following format:**

Question: 
[Rewrite the user's question into a clear legal question {query}]

Referenced Article:
[Specify the article number(s) referenced in the answer]

Simplified Answer: 
[provide a clear and concise simplified explanation of the legal answer]
"""
    
    final_response_obj = await asyncio.to_thread(
        answer_llm.generate_content,
        final_prompt
    )
    response_text = final_response_obj.text.strip()
    
    # تحليل الإجابة
    parts = {
        "question": query,
        "legal_answer": "",
        "referenced_article": "",
        "simplified": ""
    }
    
    lines = response_text.split("\n")
    current_section = None
    
    for line in lines:
        line = line.strip()
        if line.startswith("Question:"):
            current_section = "question"
            parts["question"] = line.replace("Question:", "").strip()
        elif line.startswith("Legal Answer:"):
            current_section = "legal_answer"
        elif line.startswith("Referenced Article:"):
            current_section = "referenced_article"
        elif line.startswith("Simplified Answer:"):
            current_section = "simplified"
        elif current_section and line:
            if parts[current_section]:
                parts[current_section] += " " + line
            else:
                parts[current_section] = line
    
    return parts

# ========== API Endpoints ==========
@app.get("/")
async def root():
    """الصفحة الرئيسية للـ API"""
    return {
        "message": "Legal RAG API is running! 🚀",
        "version": "1.0.0",
        "endpoints": {
            "search": "/api/search",
            "health": "/api/health"
        },
        "docs": "/docs"
    }

@app.get("/api/health")
async def health_check():
    """فحص صحة النظام"""
    return {
        "status": "healthy",
        "models_loaded": True,
        "database_records": len(df)
    }

@app.post("/api/search", response_model=SearchResponse)
async def search_legal_query(request: QueryRequest):
    """
    معالجة استعلام قانوني كامل
    
    Args:
        request: القاموس يحتوي على query (السؤال)
    
    Returns:
        نتائج البحث والإجابة الكاملة
    """
    try:
        query = request.query.strip()
        
        if not query:
            raise HTTPException(status_code=400, detail="السؤال فارغ")
        
        # 1) إعادة صياغة السؤال
        rewritten = await rewrite_query(query)
        cleaned = rewritten.replace(" ", "")
        
        # 2) تحديد نوع البحث
        articles = []
        search_type = "semantic"
        
        if cleaned.isdigit():
            # بحث مباشر
            search_type = "direct"
            articles = await search_direct_article(cleaned)
            
            if not articles:
                raise HTTPException(
                    status_code=404,
                    detail=f"المادة {cleaned} غير موجودة"
                )
        else:
            # بحث دلالي
            articles = await search_semantic(rewritten)
            
            if not articles:
                raise HTTPException(
                    status_code=404,
                    detail="لم يتم العثور على نتائج"
                )
        
        # 3) توليد الإجابة
        answer = await generate_answer(query, articles)
        
        return SearchResponse(
            rewritten_query=rewritten,
            articles=[ArticleResponse(**art) for art in articles],
            answer=AnswerResponse(**answer),
            search_type=search_type
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"خطأ في المعالجة: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    print("🚀 بدء تشغيل السيرفر على http://localhost:8000")
    print("📚 وثائق API: http://localhost:8000/docs")
    # uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)