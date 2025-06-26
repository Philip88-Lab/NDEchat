import streamlit as st
import faiss
import pickle
import numpy as np
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

import torch

# ---- SIDEBAR ----
st.sidebar.title("Settings")
GEMINI_API_KEY = st.sidebar.text_input("Gemini API Key", type="password")
TOP_K = st.sidebar.number_input("Number of results (K)", min_value=1, max_value=10, value=3, step=1)

st.title("🔎 NDE (Near-Death Experience) Retrieval Chatbot")
st.write("Ask any question about NDEs! The chatbot retrieves real case evidence and answers with Gemini.")

# ---- LOAD FAISS INDEX & MAPPING ----
@st.cache_resource(show_spinner="Loading index and mapping...")
def load_index_and_mapping():
    index = faiss.read_index("nde_faiss.index")
    with open("nde_doc_mapping.pkl", "rb") as f:
        doc_map = pickle.load(f)
    return index, doc_map

index, document_id_to_original_doc = load_index_and_mapping()

# ---- LOAD EMBEDDING MODEL ----
@st.cache_resource(show_spinner="Loading embedding model...")
def load_embedding_model():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'},
        encode_kwargs={'normalize_embeddings': False}
    )

embedding_model = load_embedding_model()

# ---- Gemini LLM ----
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

def retrieve_relevant_documents(query, k_results=3):
    query_emb = embedding_model.embed_query(query)
    D, I = index.search(np.array([query_emb]).astype('float32'), k_results)
    return [document_id_to_original_doc[idx] for idx in I[0] if idx in document_id_to_original_doc]

def format_docs_for_context(docs):
    out = []
    for doc in docs:
        cid = doc.metadata.get('id', '')
        title = doc.metadata.get('title', '')
        snippet = doc.page_content[:500]
        out.append(f"Source ID: {cid}\nURL: {title}\nExcerpt:\n{snippet}")
    return "\n\n===\n\n".join(out)

PROMPT_TEMPLATE = """You are an assistant specializing in Near-Death Experience (NDE) studies. Answer questions strictly based on the following context. If the answer requires specific case reference, always cite the Source ID. Do not make up information.

Context:
{context}

Question: {question}

Answer:
"""
prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

# ---- STREAMLIT APP LOGIC ----
if "messages" not in st.session_state:
    st.session_state["messages"] = []

user_query = st.text_area("Your question:", height=100)
submit = st.button("Ask")

if submit and user_query.strip():
    st.session_state["messages"].append({"role": "user", "content": user_query})

    with st.spinner("Retrieving and generating answer..."):
        # Retrieval
        retrieved_docs = retrieve_relevant_documents(user_query, k_results=TOP_K)
        context = format_docs_for_context(retrieved_docs)

        # LLM generation
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

    # Display
    st.session_state["messages"].append({"role": "assistant", "content": answer})
    st.markdown("#### Chatbot Answer")
    st.write(answer)

    with st.expander("Retrieved case context (for transparency)", expanded=False):
        st.code(context)

# Display previous messages
if st.session_state["messages"]:
    st.markdown("---")
    st.markdown("### Conversation History")
    for m in st.session_state["messages"]:
        st.write(f"**{m['role'].capitalize()}:** {m['content']}")
