import streamlit as st
import os
import zipfile
import requests
import faiss
import pickle
import torch
import numpy as np
from whoosh.index import open_dir
from whoosh.qparser import QueryParser
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langdetect import detect

# --- Google Drive 公共直链，替换为你的zip文件ID ---
WHOOSH_ZIP_URL = "https://drive.google.com/uc?export=download&id=163lLNm20vBzoFdOWq4VOdD3d0FFampaL"

# --- 下载并解压 whoosh_indexdir.zip ---
def download_and_extract_zip(url, target_dir="indexdir", zip_name="whoosh_indexdir.zip"):
    if not os.path.exists(target_dir):
        st.info(f"Downloading Whoosh index zip from cloud storage...")
        r = requests.get(url)
        with open(zip_name, "wb") as f:
            f.write(r.content)
        st.info(f"Extracting {zip_name} ...")
        with zipfile.ZipFile(zip_name, 'r') as zip_ref:
            zip_ref.extractall(".")
        st.success("Whoosh index ready.")

# --- 侧栏配置 ---
st.sidebar.title("Settings")
GEMINI_API_KEY = st.sidebar.text_input("Gemini API Key", type="password")
TOP_K = st.sidebar.number_input("Number of results (K)", 1, 10, 5, 1)

st.title("🔎 NDE Retrieval Chatbot (Hybrid Search with Cloud Whoosh Index)")

# 下载并解压索引
download_and_extract_zip(WHOOSH_ZIP_URL)

# 加载whoosh索引
@st.cache_resource
def load_whoosh_index():
    return open_dir("indexdir")

whoosh_ix = load_whoosh_index()
qp = QueryParser("content", whoosh_ix.schema)

# 加载faiss索引和映射（本地需上传）
@st.cache_resource
def load_faiss_index():
    index = faiss.read_index("nde_faiss.index")
    with open("nde_doc_mapping.pkl", "rb") as f:
        doc_map_raw = pickle.load(f)
    doc_map = {i: Document(page_content=d["page_content"], metadata=d["metadata"]) for i, d in doc_map_raw.items()}
    return index, doc_map

faiss_index, faiss_docs = load_faiss_index()

# 加载embedding模型
@st.cache_resource
def load_embedding_model():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'},
        encode_kwargs={'normalize_embeddings': False}
    )

embedding_model = load_embedding_model()

def detect_language(text):
    try:
        return detect(text)
    except:
        return "en"

def load_llm():
    if not GEMINI_API_KEY:
        st.warning("Please enter your Gemini API Key.")
        st.stop()
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-flash-latest",
        temperature=0.6,
        top_p=0.95,
        google_api_key=GEMINI_API_KEY
    )

def semantic_search(query, k=5):
    query_emb = embedding_model.embed_query(query)
    _, I = faiss_index.search(np.array([query_emb]).astype('float32'), k)
    return [faiss_docs[idx] for idx in I[0] if idx in faiss_docs]

def keyword_search(query, k=5):
    with whoosh_ix.searcher() as s:
        results = s.search(qp.parse(query), limit=k)
        ids = [int(r['id']) for r in results]
        return [faiss_docs[idx] for idx in ids if idx in faiss_docs]

def format_docs(docs):
    return "\n\n---\n\n".join(
        f"ID: {d.metadata.get('id')}\nURL: {d.metadata.get('title')}\nDate: {d.metadata.get('date')}\nLang: {d.metadata.get('language')}\nContent:\n{d.page_content[:800]}"
        for d in docs
    )

user_query = st.text_area("Your question:", height=100)
submit = st.button("Ask")

if submit and user_query.strip():
    user_lang = detect_language(user_query)
    prompt_template = """You are an NDE expert. Answer based strictly on context. Cite ID clearly.

Context:
{context}

Question: {question}

Answer:""" if user_lang.startswith('en') else """你是濒死体验专家，根据上下文详细回答并注明案例ID。

上下文:
{context}

问题: {question}

回答:"""

    prompt = ChatPromptTemplate.from_template(prompt_template)

    with st.spinner("Searching..."):
        docs = keyword_search(user_query, TOP_K)
        if not docs:
            docs = semantic_search(user_query, TOP_K)

        context = format_docs(docs)
        llm = load_llm()

        rag_chain = (
            {"context": lambda _: context, "question": RunnablePassthrough()}
            | prompt | llm | StrOutputParser()
        )
        answer = rag_chain.invoke({"question": user_query})

    st.markdown("### Answer")
    st.write(answer)
    with st.expander("Retrieved context"):
        st.code(context)
