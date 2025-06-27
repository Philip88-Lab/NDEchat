import streamlit as st
import faiss, pickle, torch, re
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

# Sidebar config
st.sidebar.title("Settings")
GEMINI_API_KEY = st.sidebar.text_input("Gemini API Key", type="password")
TOP_K = st.sidebar.number_input("Number of results (K)", 1, 10, 5, 1)

st.title("🔎 NDE Retrieval Chatbot (Hybrid Search)")

# Load indexes
@st.cache_resource
def load_faiss():
    index = faiss.read_index("nde_faiss.index")
    with open("nde_doc_mapping.pkl", "rb") as f:
        doc_map_raw = pickle.load(f)
    doc_map = {i: Document(page_content=d["page_content"], metadata=d["metadata"]) for i, d in doc_map_raw.items()}
    return index, doc_map

@st.cache_resource
def load_whoosh():
    return open_dir("indexdir")

faiss_index, faiss_docs = load_faiss()
whoosh_ix = load_whoosh()
qp = QueryParser("content", whoosh_ix.schema)

# Load embedding model
@st.cache_resource
def load_embedding_model():
    return HuggingFaceEmbeddings("sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'})

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
    return ChatGoogleGenerativeAI("gemini-1.5-flash-latest", temperature=0.6, top_p=0.95, google_api_key=GEMINI_API_KEY)

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
        # Hybrid logic: keyword first, then semantic if no match
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

