import streamlit as st
import os
import numpy as np
import faiss
import pickle
from langchain_core.documents import Document # 需要用于反序列化 Document 对象 (如果映射包含完整对象)
# from google.colab import userdata # 在 Streamlit Cloud 上不需要，API Key 通过 Secrets 管理

# --- LangChain 和模型导入 ---
# (这些与您 RAG 管道中的导入相同)
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# --- 配置 ---
st.set_page_config(page_title="NDE 查询机器人", layout="wide")
st.title("NDE (濒死体验) 查询机器人")

# --- 加载预计算的资源 (FAISS 索引, 文档映射) 和模型 ---
@st.cache_resource # Streamlit 缓存装饰器，避免重复加载
def load_resources():
    print("Loading resources for Streamlit app...")
    # 1. 加载 FAISS 索引
    faiss_index_path = "nde_faiss.index" # 假设与 app.py 在同一目录或指定完整路径
    if not os.path.exists(faiss_index_path):
        st.error(f"FAISS index file not found at {faiss_index_path}. Please ensure it's available.")
        return None, None, None, None
    try:
        loaded_index = faiss.read_index(faiss_index_path)
        print(f"FAISS index loaded successfully from {faiss_index_path} with {loaded_index.ntotal} vectors.")
    except Exception as e:
        st.error(f"Error loading FAISS index: {e}")
        return None, None, None, None

    # 2. 加载文档映射
    doc_mapping_path = "nde_doc_mapping.pkl"
    if not os.path.exists(doc_mapping_path):
        st.error(f"Document mapping file not found at {doc_mapping_path}. Please ensure it's available.")
        return loaded_index, None, None, None # 返回已加载的索引，以防部分成功
    try:
        with open(doc_mapping_path, 'rb') as f:
            loaded_doc_mapping = pickle.load(f)
        print(f"Document mapping loaded successfully with {len(loaded_doc_mapping)} entries.")
    except Exception as e:
        st.error(f"Error loading document mapping: {e}")
        return loaded_index, None, None, None # 返回已加载的索引，以防部分成功

    # 3. 初始化嵌入模型 (HuggingFace)
    try:
        hf_embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        print("HuggingFace embedding model initialized.")
    except Exception as e:
        st.error(f"Error initializing HuggingFace embedding model: {e}")
        return loaded_index, loaded_doc_mapping, None, None # 返回已加载的资源

    # 4. 初始化 LLM (Gemini)
    try:
        # 在 Streamlit 中，从 Secrets 获取 API Key
        # 注意: 部署到 Streamlit Cloud 时，需要在其 Secrets 管理中设置 GOOGLE_API_KEY
        gemini_api_key_st = os.environ.get('GOOGLE_API_KEY') # 部署时从环境变量获取

        if not gemini_api_key_st:
            # 在 Streamlit Cloud 上，如果 Secret 未设置，os.environ.get 会返回 None
            # 不需要尝试 google.colab.userdata
            st.warning("GOOGLE_API_KEY not found in environment secrets. LLM will likely fail if a key is required.")
            llm_model = None # 或者根据需要决定是否让应用在这里停止
        else:
            llm_model = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash-latest",
                google_api_key=gemini_api_key_st,
                temperature=0.7
            )
            print("Gemini LLM initialized.")

    except Exception as e:
        st.error(f"Error initializing Gemini LLM: {e}")
        llm_model = None

    return loaded_index, loaded_doc_mapping, hf_embedding_model, llm_model

# 加载资源
index_st, doc_mapping_st, embedding_model_st, llm_st = load_resources()

# --- RAG 管道组件 (与 Notebook 中的定义类似) ---
def format_docs_for_context_st(docs: list[Document]) -> str:
    formatted_strings = []
    for i, doc_obj in enumerate(docs):
        source_id = doc_obj.metadata.get('id', f'Unknown Source ID {i}')
        content_snippet = doc_obj.page_content
        # Streamlit markdown 需要双反斜杠表示换行，或者直接使用包含换行符的f-string
        formatted_strings.append(f"Source ID: {source_id}\nContent:\n{content_snippet}")
    return "\n\n===\n\n".join(formatted_strings)

def retrieve_relevant_documents_st(query: str, k_results: int = 3) -> list[Document]:
    if index_st is None or not doc_mapping_st or embedding_model_st is None:
        st.warning("Core resources (index, mapping, or embedding model) not loaded. Retrieval cannot proceed.")
        return []
    try:
        # 打印到 Streamlit 界面，方便调试
        st.write(f"--- Debugging Retrieval for query: '{query}' ---")
        st.write(f"Embedding model being used: {type(embedding_model_st)}")

        query_embedding = embedding_model_st.embed_query(query)
        # st.write(f"Query embedding (first 5 dims): {query_embedding[:5]}") # 可以取消注释查看部分嵌入向量

        query_embedding_np = np.array([query_embedding]).astype('float32')
        
        st.write(f"Searching FAISS index (type: {type(index_st)}, ntotal: {index_st.ntotal}) with k={k_results}")
        distances, indices = index_st.search(query_embedding_np, k_results)
        
        st.write(f"FAISS search raw results - Distances: {distances}, Indices: {indices}")

        retrieved_docs_details = []
        retrieved_docs_actual = []

        if indices.size > 0: # 确保 indices 不是空的
            for i, faiss_idx in enumerate(indices[0]):
                detail = {"faiss_original_index": int(faiss_idx)} # Store original FAISS index
                if faiss_idx != -1:
                    detail["distance_score"] = float(distances[0][i])
                    doc_object = doc_mapping_st.get(faiss_idx) # 从映射中获取
                    if doc_object:
                        retrieved_docs_actual.append(doc_object)
                        detail["doc_id"] = doc_object.metadata.get('id')
                        detail["content_snippet"] = doc_object.page_content[:150] + "..."
                    else:
                        detail["error"] = f"FAISS Index {faiss_idx} not found in doc_mapping_st."
                else: # faiss_idx == -1
                    detail["info"] = "FAISS returned -1 (no more similar documents for this rank)"
                    # Assign a large distance if distances array might not cover this
                    detail["distance_score"] = float(distances[0][i]) if i < distances.shape[1] else float('inf')

                retrieved_docs_details.append(detail)
        
        st.write("Retrieved documents (details for debugging):")
        st.json(retrieved_docs_details) # 用 st.json 更易读
        st.write("--- End Debugging Retrieval ---")

        return retrieved_docs_actual
    except Exception as e:
        st.error(f"检索文档时出错: {e}")
        st.exception(e) # 打印完整的异常堆栈到界面
        return []

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

# 构建 RAG 链 (如果所有组件都已加载)
rag_chain_st = None # 初始化为 None
if llm_st and index_st and doc_mapping_st and embedding_model_st and prompt_template_st:
    retriever_runnable_st = RunnableLambda(
        lambda input_dict: retrieve_relevant_documents_st(input_dict["question"], k_results=3)
    )
    rag_chain_st = (
        {
            "context": retriever_runnable_st | format_docs_for_context_st,
            "question": RunnablePassthrough() # 将整个输入字典传递下去，因为问题在里面
        }
        | prompt_template_st
        | llm_st
        | StrOutputParser()
    )
    st.success("聊天机器人已准备就绪！")
else:
    # rag_chain_st 保持为 None
    missing_components = []
    if not llm_st: missing_components.append("LLM")
    if not index_st: missing_components.append("FAISS index")
    if not doc_mapping_st: missing_components.append("Document mapping")
    if not embedding_model_st: missing_components.append("Embedding model")
    if not prompt_template_st: missing_components.append("Prompt template") # 理论上这个总会成功
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
            with st.spinner("正在思考并检索信息..."):
                try:
                    # RAG 链期望一个字典作为输入，其中包含 "question" 键
                    response = rag_chain_st.invoke({"question": user_query})
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    error_message = f"处理您的问题时发生错误: {e}"
                    st.error(error_message)
                    st.session_state.messages.append({"role": "assistant", "content": "抱歉，我无法回答您的问题。"})
        else:
            st.warning("聊天机器人未激活，无法处理查询。请检查上面的错误信息。")

st.sidebar.info("这是一个基于检索增强生成 (RAG) 的 NDE 聊天机器人原型。")
