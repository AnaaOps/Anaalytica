import os
import tempfile
import streamlit as st
from dotenv import load_dotenv
import numpy as np

# OCR imports
import pytesseract
from pdf2image import convert_from_path

from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

load_dotenv()

# -----------------------------
# OCR helper
# -----------------------------
def ocr_pdf(pdf_path: str) -> str:
    images = convert_from_path(pdf_path)
    text = ""
    for img in images:
        text += pytesseract.image_to_string(img) + "\n"
    return text.strip()

# -----------------------------
# Streamlit config
# -----------------------------
st.set_page_config(
    page_title="ANAALYTICA",
    page_icon="📄",
    layout="wide"
)

st.title("👩‍🎓 ANAALYTICA")
st.caption("Turn documents into intelligence.")

# -----------------------------
# API keys check
# -----------------------------
if not os.getenv("GROQ_API_KEY"):
    st.error("❌ GROQ_API_KEY missing (required)")
    st.stop()

if not os.getenv("OPENAI_API_KEY"):
    st.warning("⚠️ OPENAI_API_KEY not set — local embeddings will be used")

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.header("⚙ Settings")
    chunk_size = st.slider("Chunk size", 400, 1500, 800, 100)
    chunk_overlap = st.slider("Chunk overlap", 0, 300, 100, 10)

    if st.button("🧹 Clear Vector Database"):
        import shutil
        if os.path.exists("chroma_db"):
            shutil.rmtree("chroma_db")
            st.success("Vector DB cleared")

# -----------------------------
# Upload PDFs
# -----------------------------
uploaded_files = st.file_uploader(
    "Upload PDF files",
    type=["pdf"],
    accept_multiple_files=True
)

# -----------------------------
# Process PDFs (TEXT + SCANNED)
# -----------------------------
if st.button("🚀 Process PDFs") and uploaded_files:
    import shutil

    if os.path.exists("chroma_db"):
        shutil.rmtree("chroma_db")

    docs = []

    with st.spinner("📄 Reading PDFs..."):
        for pdf in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(pdf.read())

            loader = PyPDFLoader(tmp.name)
            loaded_docs = loader.load()

            # 🔍 Detect scanned PDF
            if not loaded_docs or all(len(d.page_content.strip()) == 0 for d in loaded_docs):
                st.warning(f"🖼️ Scanned PDF detected: {pdf.name} — running OCR")
                ocr_text = ocr_pdf(tmp.name)

                if ocr_text.strip():
                    loaded_docs = [
                        Document(
                            page_content=ocr_text,
                            metadata={"source": pdf.name}
                        )
                    ]
                else:
                    st.error(f"❌ OCR failed for {pdf.name}")
                    continue

            docs.extend(loaded_docs)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_documents(docs)

    # -----------------------------
    # Embeddings (OpenAI → HF → TF-IDF)
    # -----------------------------
    try:
        from langchain_openai import OpenAIEmbeddings

        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )

        vectorstore = Chroma.from_documents(
            chunks,
            embeddings,
            persist_directory="chroma_db"
        )

        st.session_state.vectorstore = vectorstore
        st.success(f"✅ Processed {len(chunks)} chunks (OpenAI embeddings)")

    except Exception as e:
        st.warning(f"OpenAI embeddings failed → {e}")

        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings

            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )

            vectorstore = Chroma.from_documents(
                chunks,
                embeddings,
                persist_directory="chroma_db"
            )

            st.session_state.vectorstore = vectorstore
            st.success(f"✅ Processed {len(chunks)} chunks (local embeddings)")

        except Exception as e2:
            st.warning("Using TF-IDF fallback")

            from sklearn.feature_extraction.text import TfidfVectorizer

            texts = [c.page_content for c in chunks]
            metadatas = [c.metadata for c in chunks]

            st.session_state.tfidf_texts = texts
            st.session_state.tfidf_metadatas = metadatas
            st.session_state.tfidf_vectorizer = TfidfVectorizer(max_features=512)
            st.session_state.tfidf_matrix = st.session_state.tfidf_vectorizer.fit_transform(texts)

            st.success(f"✅ Processed {len(chunks)} chunks (TF-IDF)")

# -----------------------------
# Question Answering
# -----------------------------
if "vectorstore" in st.session_state or "tfidf_texts" in st.session_state:
    st.divider()
    question = st.text_input("💬 Ask a question about the PDFs")

    if question:
        try:
            from langchain_groq import ChatGroq

            llm = ChatGroq(
                model="llama-3.3-70b-versatile",
                temperature=0,
                api_key=os.getenv("GROQ_API_KEY")
            )

            use_mock = False
        except Exception as e:
            st.error(f"LLM failed: {e}")
            use_mock = True

        if use_mock:
            class MockLLM:
                def invoke(self, prompt):
                    class R:
                        content = "LLM unavailable"
                    return R()
            llm = MockLLM()

        # Retrieve docs
        relevant_docs = []

        if "vectorstore" in st.session_state:
            relevant_docs = st.session_state.vectorstore.similarity_search(question, k=4)
        else:
            from sklearn.metrics.pairwise import cosine_similarity

            q_vec = st.session_state.tfidf_vectorizer.transform([question])
            sims = cosine_similarity(q_vec, st.session_state.tfidf_matrix).flatten()
            top = np.argsort(sims)[-4:][::-1]

            for i in top:
                relevant_docs.append(
                    Document(
                        page_content=st.session_state.tfidf_texts[i],
                        metadata=st.session_state.tfidf_metadatas[i]
                    )
                )

        context = "\n\n".join(d.page_content for d in relevant_docs)

        prompt = f"""
Answer using ONLY the context below.
If not present, say "I don't know".

Context:
{context}

Question:
{question}
"""

        with st.spinner("🧠 Thinking..."):
            response = llm.invoke(prompt)

        st.subheader("🧠 Answer")
        st.write(getattr(response, "content", response))

        st.subheader("📚 Sources")
        for d in relevant_docs:
            src = d.metadata.get("source", "PDF")
            page = d.metadata.get("page", "N/A")
            st.write(f"- {src} (page {page})")

else:
    st.info("📤 Upload PDFs and click **Process PDFs**")
