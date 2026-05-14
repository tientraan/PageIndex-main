import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from pypdf import PdfReader
from docx import Document
import tempfile
from google import genai
import os
from dotenv import load_dotenv

# ======================================
# CẤU HÌNH & KHỞI TẠO
# ======================================
load_dotenv()

st.set_page_config(
    page_title="InsightDocs AI",
    page_icon="🧠",
    layout="wide"
)

# Custom CSS để giao diện mượt mà hơn
st.markdown("""
    <style>
    .stApp { background-color: #f8f9fa; }
    .main-header { font-size: 2.5rem; font-weight: 800; color: #1E1E1E; margin-bottom: 0.5rem; }
    .sub-header { color: #555; margin-bottom: 2rem; }
    </style>
""", unsafe_allow_html=True)

# Khởi tạo Client Gemini
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

# Load model embedding (Cache để tiết kiệm RAM)
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedding_model = load_embedding_model()

# ======================================
# HÀM XỬ LÝ DỮ LIỆU
# ======================================

def read_file(file):
    suffix = os.path.splitext(file.name)[1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file.read())
        path = tmp.name

    text = ""
    if suffix == ".pdf":
        reader = PdfReader(path)
        text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    elif suffix == ".docx":
        doc = Document(path)
        text = "\n".join([p.text for p in doc.paragraphs])
    else:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
    
    os.remove(path)
    return text

def split_text(text, chunk_size=800):
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i + chunk_size]
        if chunk.strip(): chunks.append(chunk)
    return chunks

# ======================================
# GIAO DIỆN THANH BÊN (SIDEBAR)
# ======================================

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/281/281764.png", width=80) # Icon Gemini hoặc tùy chọn
    st.title("⚙️ Cấu hình App")
    
    st.markdown("---")
    uploaded_file = st.file_uploader("📁 Tải tài liệu lên", type=["pdf", "docx", "txt"])
    
    top_k = st.slider("Số lượng đoạn trích tìm kiếm (Top K)", 1, 10, 3)
    
    if st.button("🗑️ Xóa lịch sử Chat"):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.caption("Powered by Gemini 2.5 Flash & FAISS")

# ======================================
# XỬ LÝ LOGIC CHÍNH
# ======================================

# Khởi tạo session state cho lịch sử chat và dữ liệu vector
if "messages" not in st.session_state:
    st.session_state.messages = []

if uploaded_file:
    # Chỉ xử lý file nếu là file mới
    if "current_file" not in st.session_state or st.session_state.current_file != uploaded_file.name:
        with st.status(" đang xử lý tài liệu...", expanded=True) as status:
            st.write("🔍 Đang đọc nội dung...")
            text = read_file(uploaded_file)
            
            st.write("🧩 Đang phân tách văn bản...")
            chunks = split_text(text)
            
            st.write("⚡ Đang tạo chỉ mục vector...")
            embeddings = embedding_model.encode(chunks)
            dimension = embeddings.shape[1]
            index = faiss.IndexFlatL2(dimension)
            index.add(np.array(embeddings).astype("float32"))
            
            # Lưu vào session state để dùng lại
            st.session_state.chunks = chunks
            st.session_state.index = index
            st.session_state.current_file = uploaded_file.name
            status.update(label="✅ Đã xử lý xong tài liệu!", state="complete", expanded=False)

# ======================================
# GIAO DIỆN CHAT CHÍNH
# ======================================

st.markdown('<p class="main-header">📚 InsightDocs AI Search</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Trợ lý ảo phân tích tài liệu thông minh sử dụng Gemini 2.5 Flash</p>', unsafe_allow_html=True)

# Hiển thị lịch sử chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Nhận câu hỏi từ người dùng
if prompt := st.chat_input("Nhập câu hỏi của bạn về tài liệu..."):
    if not uploaded_file:
        st.error("Vui lòng tải tài liệu lên ở thanh bên trái trước!")
    else:
        # 1. Hiển thị câu hỏi người dùng
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 2. Xử lý phản hồi từ AI
        with st.chat_message("assistant"):
            with st.spinner("AI đang tìm kiếm thông tin..."):
                # Tìm kiếm vector
                query_embedding = embedding_model.encode([prompt])
                distances, indices = st.session_state.index.search(
                    np.array(query_embedding).astype("float32"), top_k
                )
                
                contexts = [st.session_state.chunks[idx] for idx in indices[0] if 0 <= idx < len(st.session_state.chunks)]
                context_text = "\n\n".join(contexts)

                # Gọi Gemini
                full_prompt = f"Dựa vào tài liệu này:\n{context_text}\n\nCâu hỏi: {prompt}\nTrả lời rõ ràng bằng tiếng Việt."
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=full_prompt
                )
                answer = response.text

                # Hiển thị và lưu phản hồi
                st.markdown(answer)
                
                with st.expander("📌 Xem các đoạn trích nguồn được sử dụng"):
                    for i, c in enumerate(contexts):
                        st.info(f"Nguồn {i+1}: {c}")

        st.session_state.messages.append({"role": "assistant", "content": answer})