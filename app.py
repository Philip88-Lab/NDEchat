import streamlit as st
import os
import numpy as np
import faiss
import pickle
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
import traceback # 导入 traceback 模块
import time      # 导入 time 模块

# --- 配置 ---
st.set_page_config(page_title="NDE 查询机器人", layout="wide")
st.title("NDE (濒死体验) 查询机器人")

# --- 加载预计算的资源 (FAISS 索引, 文档映射) 和模型 ---
@st.cache_resource # Streamlit 缓存装饰器，避免重复加载
def load_resources():
    print(">>> LOAD_RESOURCES: Attempting to load resources...")

    loaded_index = None
    loaded_doc_mapping = None
    hf_embedding_model = None
    llm_model = None

    # 1. 加载 FAISS 索引
    faiss_index_path = "nde_faiss.index"
    if not os.path.exists(faiss_index_path):
        print(f">>> LOAD_RESOURCES: FAISS index file NOT FOUND at {faiss_index_path}")
        st.error(f"FAISS index file not found at {faiss_index_path}. Please ensure it's available.")
    else:
        try:
            loaded_index = faiss.read_index(faiss_index_path)
            print(f">>> LOAD_RESOURCES: FAISS index loaded successfully. Vectors: {loaded_index.ntotal if loaded_index else 'None'}")
        except Exception as e_faiss:
            print(f">>> LOAD_RESOURCES: ERROR loading FAISS index: {str(e_faiss)}\n{traceback.format_exc()}")
            st.error(f"Error loading FAISS index: {e_faiss}")

    # 2. 加载文档映射
    doc_mapping_path = "nde_doc_mapping.pkl"
    if not os.path.exists(doc_mapping_path):
        print(f">>> LOAD_RESOURCES: Document mapping file NOT FOUND at {doc_mapping_path}")
        st.error(f"Document mapping file not found at {doc_mapping_path}. Please ensure it's available.")
    else:
        try:
            with open(doc_mapping_path, 'rb') as f:
                loaded_doc_mapping = pickle.load(f)
            print(f">>> LOAD_RESOURCES: Document mapping loaded successfully. Entries: {len(loaded_doc_mapping) if loaded_doc_mapping else 'None'}")
        except Exception as e_map:
            print(f">>> LOAD_RESOURCES: ERROR loading document mapping: {str(e_map)}\n{traceback.format_exc()}")
            st.error(f"Error loading document mapping: {e_map}")

    # 3. 初始化嵌入模型 (HuggingFace)
    try:
        hf_embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        print(">>> LOAD_RESOURCES: HuggingFace embedding model INITIALIZED successfully.")
    except Exception as e_hf:
        print(f">>> LOAD_RESOURCES: ERROR initializing HuggingFace embedding model: {str(e_hf)}\n{traceback.format_exc()}")
        st.error(f"Error initializing HuggingFace embedding model: {e_hf}")
        hf_embedding_model = None

    # 4. 初始化 LLM (Gemini)
    try:
        gemini_api_key_st = os.environ.get('GOOGLE_API_KEY')
        print(f">>> LOAD_RESOURCES: GOOGLE_API_KEY from env: '{'SET' if gemini_api_key_st else 'NOT_SET'}'")
        if not gemini_api_key_st:
            st.warning("GOOGLE_API_KEY not found in environment secrets. LLM will likely fail if a key is required.")
            llm_model = None
        else:
            llm_model = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash-latest",
                google_api_key=gemini_api_key_st,
                temperature=0.7
            )
            print(">>> LOAD_RESOURCES: Gemini LLM INITIALIZED successfully.")
    except Exception as e_llm:
        print(f">>> LOAD_RESOURCES: ERROR initializing Gemini LLM: {str(e_llm)}\n{traceback.format_exc()}")
        st.error(f"Error initializing Gemini LLM: {e_llm}")
        llm_model = None

    print(">>> LOAD_RESOURCES: Finished loading resources.")
    return loaded_index, loaded_doc_mapping, hf_embedding_model, llm_model

# 加载资源
try:
    index_st, doc_mapping_st, embedding_model_st, llm_st = load_resources()
except Exception as e_load_res_call:
    st.error(f"CRITICAL ERROR during the call to load_resources: {e_load_res_call}")
    st.text(traceback.format_exc())
    index_st, doc_mapping_st, embedding_model_st, llm_st = None, None, None, None


# --- RAG 管道组件 ---
def format_docs_for_context_st(docs: list[Document]) -> str:
    print(f">>> FORMAT_DOCS: Formatting {len(docs)} documents for context.")
    formatted_strings = []
    for i, doc_obj in enumerate(docs):
        source_id = doc_obj.metadata.get('id', f'Unknown Source ID {i}')
        content_snippet = doc_obj.page_content
        formatted_strings.append(f"Source ID: {source_id}\nContent:\n{content_snippet}")
    return "\n\n===\n\n".join(formatted_strings)

def retrieve_relevant_documents_st(query: str, k_results: int = 3) -> list[Document]: # Default k=3
    print(f">>> RETRIEVE_FUNC: CALLED with query: '{query}', k_results_param: {k_results} at {time.time()}")

    if index_st is None:
        print(">>> RETRIEVE_FUNC: ERROR - FAISS index (index_st) is None!")
        return []
    if not doc_mapping_st:
        print(">>> RETRIEVE_FUNC: ERROR - Document mapping (doc_mapping_st) is empty or None!")
        return []
    if embedding_model_st is None:
        print(">>> RETRIEVE_FUNC: ERROR - Embedding model (embedding_model_st) is None!")
        return []

    retrieved_docs_actual = []
    try:
        print(f">>> RETRIEVE_FUNC: Embedding model type: {type(embedding_model_st)}")
        query_embedding = None
        try:
            print(f">>> RETRIEVE_FUNC: Attempting to embed query: '{query}' at {time.time()}")
            query_embedding = embedding_model_st.embed_query(query)
            print(f">>> RETRIEVE_FUNC: Query embedded successfully at {time.time()}. Embedding type: {type(query_embedding)}, Len: {len(query_embedding) if hasattr(query_embedding, '__len__') else 'N/A'}")
        except Exception as e_embed:
            print(f">>> RETRIEVE_FUNC: ERROR during query embedding: {str(e_embed)}\n{traceback.format_exc()}")
            return []

        if query_embedding is None:
            print(">>> RETRIEVE_FUNC: ERROR - Query embedding resulted in None.")
            return []

        query_embedding_np = np.array([query_embedding]).astype('float32')
        print(f">>> RETRIEVE_FUNC: Query embedding NumPy shape: {query_embedding_np.shape}")

        distances, indices = None, None
        try:
            print(f">>> RETRIEVE_FUNC: Attempting FAISS search. Index ntotal: {index_st.ntotal if index_st else 'None'}, k: {k_results}")
            if index_st:
                 distances, indices = index_st.search(query_embedding_np, k_results)
                 print(f">>> RETRIEVE_FUNC: FAISS search completed. Indices: {indices}, Distances: {distances}")
            else:
                 print(">>> RETRIEVE_FUNC: CRITICAL - index_st became None before search!")
                 return []
        except Exception as e_search:
            print(f">>> RETRIEVE_FUNC: ERROR during FAISS search: {str(e_search)}\n{traceback.format_exc()}")
            return []

        if indices is None or indices.size == 0:
            print(">>> RETRIEVE_FUNC: FAISS search returned no indices.")
            return []
            
        retrieved_docs_details_for_log = []
        for i, faiss_idx_np in enumerate(indices[0]):
            faiss_idx = int(faiss_idx_np)
            detail = {"faiss_original_index": faiss_idx}
            current_distance = float('inf')
            if distances is not None and i < distances.shape[1]: 
                 current_distance = float(distances[0][i])
            detail["distance_score"] = current_distance

            if faiss_idx != -1:
                doc_object = doc_mapping_st.get(faiss_idx) 
                if doc_object:
                    retrieved_docs_actual.append(doc_object)
                    detail["doc_id"] = doc_object.metadata.get('id')
                else:
                    detail["error"] = f"FAISS Index {faiss_idx} not found in doc_mapping_st."
            else:
                detail["info"] = "FAISS returned -1"
            retrieved_docs_details_for_log.append(detail)
        
        print(f">>> RETRIEVE_FUNC: Retrieved documents details (for log): {retrieved_docs_details_for_log}")

    except Exception as e_main_retrieve:
        print(f">>> RETRIEVE_FUNC: UNEXPECTED ERROR in main try-block: {str(e_main_retrieve)}\n{traceback.format_exc()}")
        return []

    print(f">>> RETRIEVE_FUNC: RETURNING {len(retrieved_docs_actual)} documents at {time.time()}")
    return retrieved_docs_actual

prompt_template_st_str = """
你是一个乐于助人的助手，专门回答有关濒死体验 (NDE) 的问题。
请严格根据下面提供的上下文信息来回答问题。
如果上下文中没有足够的信息来回答问题，或者问题与上下文无关，请明确说明你无法根据所提供的信息找到答案，不要试图编造。
请用中文回答。
在你的回答中，如果信息来自特定的记录，请务必使用以下格式引用来源记录的 ID：(来源 ID: [实际的ID号])。

上下文:
{context}

问题: {question}

回答:
"""
prompt_template_st = ChatPromptTemplate.from_template(prompt_template_st_str)

# 构建 RAG 链
rag_chain_st = None
print(">>> APP_SETUP: Attempting to build RAG chain...")
if llm_st and index_st and doc_mapping_st and embedding_model_st and prompt_template_st:
    try:
        retriever_runnable_st = RunnableLambda(
            # k_results set back to 3 (or your preferred default)
            lambda input_dict: retrieve_relevant_documents_st(input_dict["question"], k_results=3)
        )
        rag_chain_st = (
            {
                "context": retriever_runnable_st | format_docs_for_context_st,
                "question": RunnablePassthrough()
            }
            | prompt_template_st
            | llm_st
            | StrOutputParser()
        )
        print(">>> APP_SETUP: RAG chain built successfully.")
        st.success("聊天机器人已准备就绪！")
    except Exception as e_rag_build:
        print(f">>> APP_SETUP: ERROR building RAG chain: {str(e_rag_build)}\n{traceback.format_exc()}")
        st.error(f"Error building RAG chain: {e_rag_build}")
        rag_chain_st = None
else:
    missing_components = []
    if not llm_st: missing_components.append("LLM")
    if not index_st: missing_components.append("FAISS index")
    if not doc_mapping_st: missing_components.append("Document mapping")
    if not embedding_model_st: missing_components.append("Embedding model")
    print(f">>> APP_SETUP: RAG chain NOT built. Missing components: {', '.join(missing_components)}")
    st.error(f"聊天机器人未能成功初始化。缺少以下组件: {', '.join(missing_components)}. 请检查资源加载错误和API密钥。")

# --- Streamlit 聊天界面 ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "您好！我是 NDE 查询助手，有什么可以帮您的吗？"}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_query := st.chat_input("请输入您关于 NDE 的问题..."):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        if rag_chain_st:
            print(f">>> UI_CHAT: Attempting to invoke rag_chain_st with query: '{user_query}' at {time.time()}")
            try:
                response = rag_chain_st.invoke({"question": user_query})
                print(f">>> UI_CHAT: rag_chain_st.invoke successful. Response (type: {type(response)}): '{str(response)[:100]}...' at {time.time()}")
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e_invoke:
                print(f">>> UI_CHAT: ERROR during rag_chain_st.invoke: {str(e_invoke)}\n{traceback.format_exc()}")
                error_message = f"处理您的问题时发生错误 (RAG chain invocation): {e_invoke}"
                st.error(error_message)
                st.session_state.messages.append({"role": "assistant", "content": f"抱歉，处理查询时出错: {e_invoke}"})
        else:
            print(f">>> UI_CHAT: rag_chain_st is None. Cannot process query '{user_query}'. at {time.time()}")
            st.warning("聊天机器人未激活（RAG链为None），无法处理查询。请检查应用启动时的错误信息。")

st.sidebar.info("这是一个基于检索增强生成 (RAG) 的 NDE 聊天机器人原型。")
