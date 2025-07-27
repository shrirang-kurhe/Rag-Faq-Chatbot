import streamlit as st
import time
import os

from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings  # ✅ Updated import

# ------------------ Page Config ------------------
st.set_page_config(page_title="FAQ Chatbot", layout="wide")
st.title("🤖 Context-Aware FAQ Chatbot with RAG and Memory")

# ------------------ API Key & LLM ------------------
from dotenv import load_dotenv
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="Gemma2-9b-it")

# ------------------ Embedding Model ------------------
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# ------------------ Load and Cache Vectorstore ------------------
@st.cache_resource
def load_vectorstore():
    file_path = os.path.join("Context-Aware FAQ Chatbot with LangGraph and RAG", "faqs.txt")
    docs = TextLoader(file_path).load()
    chunks = CharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_documents(docs)
    return FAISS.from_documents(chunks, embedding_model)

vectorstore = load_vectorstore()

# ------------------ Topic Classifier ------------------
def classify_topic(msg: str) -> str:
    msg = msg.lower()
    if "stock" in msg or "market" in msg:
        return "Finance"
    elif "diet" in msg or "health" in msg:
        return "Health"
    elif "python" in msg or "code" in msg or "machine learning" in msg:
        return "Technology"
    else:
        return "General"

# ------------------ Generate Response ------------------
def generate_response(user_input: str) -> str:
    topic = classify_topic(user_input)
    retriever = vectorstore.as_retriever()
    docs = retriever.invoke(user_input)
    context = "\n".join([doc.page_content for doc in docs])

    history_prompt = ""
    if len(st.session_state.chat_history) >= 2:
        role1, msg1 = st.session_state.chat_history[-2]
        role2, msg2 = st.session_state.chat_history[-1]
        history_prompt = f"{role1.capitalize()}: {msg1}\n{role2.capitalize()}: {msg2}\n"

    prompt = (
        f"You are a helpful assistant specialized in {topic}.\n"
        f"Here is the past conversation:\n{history_prompt}"
        f"Context from documents:\n{context}\n"
        f"User's latest question: {user_input}"
    )

    response = llm.invoke([("user", prompt)])
    return response.content

# ------------------ Initialize Chat History ------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ------------------ Sidebar ------------------
with st.sidebar:
    st.header("🛠️ Options")
    if st.button("🧹 Clear Chat"):
        st.session_state.chat_history = []

# ------------------ Chat Interface ------------------
st.markdown("### 💬 Chat")
user_input = st.text_input("You:", placeholder="Type your question and press Enter")

if user_input:
    st.session_state.chat_history.append(("user", user_input))

    with st.spinner("Thinking..."):
        time.sleep(0.3)
        response = generate_response(user_input)

    st.session_state.chat_history.append(("assistant", response))

# ------------------ Show Chat History ------------------
st.markdown("---")
st.markdown("### 🧠 Conversation History")
for role, msg in st.session_state.chat_history:
    if role == "user":
        st.markdown(f"**🧑 You:** {msg}")
    else:
        st.markdown(f"**🤖 Bot:** {msg}")
