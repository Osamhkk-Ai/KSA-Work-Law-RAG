import pandas as pd
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
import os
from langchain_huggingface import HuggingFaceEmbeddings


load_dotenv()
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
data = pd.read_csv("laws_with_QA_langchain_combined.csv") 



metadata = data[["part", "article_id","chapter","title"]].to_dict("records")


# Load model directly

embeddings = HuggingFaceEmbeddings(
    model_name="jinaai/jina-embeddings-v3",
    model_kwargs={  
        "trust_remote_code": True}
    )


vectorstore = Chroma.from_texts(
    texts=data["combined_for_embedding"].tolist(),
    metadatas=metadata,
    embedding=embeddings,
    persist_directory="./chroma_db"
)


print("✅Done! The Chroma vector store has been created and saved.")
