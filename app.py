import streamlit as st
import faiss
import pickle
import numpy as np
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langdetect import detect
import torch

# ---- SIDEBAR ----
st.sidebar.title("Settings")
GEMINI_API_KEY = st.sidebar.text_input("Gemini API Key", type="password")
TOP_K = st.sidebar.number_input("Number of results (K)", min_value=1, max_value=10, value=5, step=1)

st.title("🔎 NDE (Near-Death Experience) Retrieval Chatbot")
st.write("Ask your question about NDEs! The chatbot retrieves real case evidence and answers in the detected language (English/Chinese).")

@st.cache_resource(show_spinner="Loading index and mapping...")
def load_index_and_mapping():
    index = faiss.read_index("nde_faiss.index")
    with open("nde_doc_mapping.pkl", "rb") as f:
        doc_map_raw = pickle.load(f)
    doc_map = {i: Document(page_content=d["page_content"], metadata=d["metadata"]) for i, d in doc_map_raw.items()}
    return index, doc_map

index, document_id_to_original_doc = load_index_and_mapping()

@st.cache_resource(show_spinner="Loading embedding model...")
def load_embedding_model():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'},
        encode_kwargs={'normalize_embeddings': False}
    )

embedding_model = load_embedding_model()

def detect_language(text):
    try:
        lang = detect(text)
        return lang
    except:
        return "en"

def load_llm():
    if not GEMINI_API_KEY:
        st.warning("Please enter your Gemini API Key in the sidebar.")
        return None
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-flash-latest",
        temperature=0.6,
        top_p=0.95,
        google_api_key=GEMINI_API_KEY
    )

def retrieve_relevant_documents(query, k_results=5):
    query_emb = embedding_model.embed_query(query)
    D, I = index.search(np.array([query_emb]).astype('float32'), k_results)
    return [document_id_to_original_doc[idx] for idx in I[0] if idx in document_id_to_original_doc]

def format_docs_for_context(docs):
    out = []
    for doc in docs:
        cid = doc.metadata.get('id', '')
        title = doc.metadata.get('title', '')
        date = doc.metadata.get('date', '')
        lang = doc.metadata.get('language', '')
        snippet = doc.page_content[:800]
        out.append(
            f"Source ID: {cid}\nURL: {title}\nDate: {date}\nLang: {lang}\nExcerpt:\n{snippet}"
        )
    return "\n\n===\n\n".join(out)

if "messages" not in st.session_state:
    st.session_state["messages"] = []

user_query = st.text_area("Your question:", height=100)
submit = st.button("Ask")

if submit and user_query.strip():
    st.session_state["messages"].append({"role": "user", "content": user_query})

    user_lang = detect_language(user_query)
    if user_lang == "zh-cn" or user_lang.startswith("zh"):
        prompt_template = """你是濒死体验（NDE）领域的知识助理。请严格依据以下上下文，用中文精准、详细地回答用户问题。如需引用具体案例请标明来源ID，不要随意编造。

上下文:
{context}

问题: {question}

详细回答:
"""
    else:
        prompt_template = """
You are an assistant specializing in Near-Death Experience (NDE) studies. 
Provide a highly detailed, specific, and comprehensive answer based strictly on the following context.
When possible, summarize multiple relevant cases and include exact descriptions and details.
Always cite the Source ID for any referenced case. Do not invent any information.

Context:
{context}

Question: {question}

Detailed Answer:
"""
    prompt = ChatPromptTemplate.from_template(prompt_template)

    with st.spinner("Retrieving and generating answer..."):
        retrieved_docs = retrieve_relevant_documents(user_query, k_results=TOP_K)
        context = format_docs_for_context(retrieved_docs)
        llm = load_llm()
        if llm is None:
            st.stop()
        rag_chain = (
            {
                "context": lambda d: context,
                "question": RunnablePassthrough()
            }
            | prompt
            | llm
            | StrOutputParser()
        )
        answer = rag_chain.invoke({"question": user_query})

    st.session_state["messages"].append({"role": "assistant", "content": answer})
    st.markdown("#### Chatbot Answer")
    st.write(answer)
    with st.expander("Retrieved case context (for transparency)", expanded=True):
        st.code(context)

if st.session_state["messages"]:
    st.markdown("---")
    st.markdown("### Conversation History")
    for m in st.session_state["messages"]:
        st.write(f"**{m['role'].capitalize()}:** {m['content']}")
