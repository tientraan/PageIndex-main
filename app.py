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
# LOAD ENV
# ======================================

load_dotenv()

# ======================================
# GEMINI CONFIG
# ======================================

client = genai.Client(
    api_key=os.environ["GEMINI_API_KEY"]
)

# ======================================
# LOAD EMBEDDING MODEL
# ======================================

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

embedding_model = load_embedding_model()

# ======================================
# READ FILES
# ======================================

def read_pdf(path):
    text = ""

    reader = PdfReader(path)

    for page in reader.pages:
        content = page.extract_text()

        if content:
            text += content + "\n"

    return text


def read_docx(path):
    doc = Document(path)

    return "\n".join(
        [p.text for p in doc.paragraphs]
    )


def read_txt(path):
    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:
        return f.read()

# ======================================
# SPLIT TEXT
# ======================================

def split_text(
    text,
    chunk_size=700
):
    chunks = []

    for i in range(
        0,
        len(text),
        chunk_size
    ):
        chunk = text[
            i:i + chunk_size
        ]

        if chunk.strip():
            chunks.append(chunk)

    return chunks

# ======================================
# CREATE VECTOR INDEX
# ======================================

def create_index(chunks):
    embeddings = embedding_model.encode(
        chunks
    )

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(
        dimension
    )

    index.add(
        np.array(
            embeddings
        ).astype("float32")
    )

    return index

# ======================================
# SEARCH RELEVANT CHUNKS
# ======================================

def search(
    query,
    index,
    chunks,
    top_k=3
):
    query_embedding = embedding_model.encode(
        [query]
    )

    distances, indices = index.search(
        np.array(
            query_embedding
        ).astype("float32"),
        top_k
    )

    results = []

    for idx in indices[0]:
        if 0 <= idx < len(chunks):
            results.append(
                chunks[idx]
            )

    return results

# ======================================
# ASK GEMINI
# ======================================

def ask_llm(
    question,
    contexts
):
    context_text = "\n\n".join(
        contexts
    )

    prompt = f"""
Bạn là AI hỗ trợ tìm kiếm tài liệu.

Dựa vào tài liệu dưới đây để trả lời câu hỏi.

TÀI LIỆU:
{context_text}

CÂU HỎI:
{question}

Hãy trả lời rõ ràng bằng tiếng Việt.
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    return response.text

# ======================================
# STREAMLIT UI
# ======================================

st.set_page_config(
    page_title="PageIndex AI",
    layout="wide"
)

st.title(
    "📚 PageIndex AI Search"
)

st.write(
    "Upload tài liệu và hỏi AI bằng Gemini 2.5 Flash"
)

# ======================================
# FILE UPLOAD
# ======================================

uploaded_file = st.file_uploader(
    "📄 Upload tài liệu",
    type=["pdf", "docx", "txt"]
)

# ======================================
# QUESTION BOX
# ======================================

question = st.text_input(
    "💬 Hỏi nội dung tài liệu"
)

# ======================================
# IF NO FILE
# ======================================

if not uploaded_file:
    st.warning(
        "Vui lòng upload tài liệu trước khi hỏi"
    )

# ======================================
# IF FILE EXISTS
# ======================================

else:
    suffix = os.path.splitext(
        uploaded_file.name
    )[1].lower()

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix
    ) as tmp:
        tmp.write(
            uploaded_file.read()
        )

        temp_path = tmp.name

    # ======================================
    # READ FILE
    # ======================================

    text = ""

    if suffix == ".pdf":
        text = read_pdf(
            temp_path
        )

    elif suffix == ".docx":
        text = read_docx(
            temp_path
        )

    elif suffix == ".txt":
        text = read_txt(
            temp_path
        )

    if not text.strip():
        st.error(
            "Không đọc được nội dung tài liệu. Vui lòng thử file khác."
        )
        st.stop()

    st.success(
        "✅ Đọc tài liệu thành công"
    )

    # ======================================
    # SPLIT TEXT
    # ======================================

    chunks = split_text(text)

    if not chunks:
        st.error(
            "Không tách được nội dung tài liệu."
        )
        st.stop()

    # ======================================
    # CREATE INDEX
    # ======================================

    index = create_index(
        chunks
    )

    st.info(
        f"📚 Đã index {len(chunks)} đoạn văn"
    )

    # ======================================
    # ASK QUESTION
    # ======================================

    if question:
        with st.spinner(
            "🤖 Gemini đang suy nghĩ..."
        ):
            contexts = search(
                question,
                index,
                chunks
            )

            answer = ask_llm(
                question,
                contexts
            )

        # ======================================
        # ANSWER
        # ======================================

        st.subheader(
            "🤖 Trả lời"
        )

        st.write(answer)

        # ======================================
        # CONTEXTS
        # ======================================

        st.subheader(
            "📄 Nội dung liên quan"
        )

        for i, c in enumerate(
            contexts
        ):
            with st.expander(
                f"Đoạn liên quan {i + 1}"
            ):
                st.write(c)