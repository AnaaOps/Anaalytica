import os
import tempfile
import streamlit as st
from dotenv import load_dotenv
import numpy as np
load_dotenv()

from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

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
    st.error("❌ GROQ_API_KEY missing (required for ChatGroq)")
    st.stop()

if not os.getenv("OPENAI_API_KEY"):
    st.warning("OPENAI_API_KEY not set — the app will use local embeddings instead of OpenAI.")

# -----------------------------
# Sidebar settings
# -----------------------------
with st.sidebar:
    st.header("⚙ Settings")
    chunk_size = st.slider("Chunk size", 400, 1500, 800, 100)
    chunk_overlap = st.slider("Chunk overlap", 0, 300, 100, 10)
    
    if st.button("🧹 Clear Vector Database"):
        import shutil
        if os.path.exists("chroma_db"):
            shutil.rmtree("chroma_db")
            st.success("Cleared ChromaDB database")
        else:
            st.info("No ChromaDB database found")

# -----------------------------
# Upload PDFs
# -----------------------------
uploaded_files = st.file_uploader(
    "Upload PDF files",
    type=["pdf"],
    accept_multiple_files=True
)

# -----------------------------
# Process PDFs
# -----------------------------
if st.button("🚀 Process PDFs") and uploaded_files:
    # Clear previous ChromaDB to prevent dimension mismatches
    import shutil
    if os.path.exists("chroma_db"):
        shutil.rmtree("chroma_db")
    
    docs = []

    with st.spinner("Reading PDFs..."):
        for pdf in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(pdf.read())
                loader = PyPDFLoader(tmp.name)
                docs.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_documents(docs)

    # Try OpenAI embeddings first
    try:
        from langchain_openai import OpenAIEmbeddings
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        
        vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory="chroma_db")
        st.session_state.vectorstore = vectorstore
        st.success(f"✅ Processed {len(chunks)} chunks (OpenAI embeddings)")

    except Exception as e:
        st.warning(f"OpenAI embeddings failed: {e}. Falling back to local embeddings.")
        
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            
            vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory="chroma_db")
            st.session_state.vectorstore = vectorstore
            st.success(f"✅ Processed {len(chunks)} chunks (local embeddings)")
            
        except Exception as e2:
            st.warning(f"Local embeddings failed: {e2}. Using simple fallback...")
            
            # Simple TF-IDF fallback
            from sklearn.feature_extraction.text import TfidfVectorizer
            import numpy as np
            
            texts = [chunk.page_content for chunk in chunks]
            metadatas = [chunk.metadata for chunk in chunks]
            
            st.session_state.tfidf_texts = texts
            st.session_state.tfidf_metadatas = metadatas
            st.session_state.tfidf_vectorizer = TfidfVectorizer(max_features=384)
            st.session_state.tfidf_matrix = st.session_state.tfidf_vectorizer.fit_transform(texts)
            
            st.success(f"✅ Processed {len(chunks)} chunks (TF-IDF embeddings)")

# -----------------------------
# Question Answering
# -----------------------------
if "vectorstore" in st.session_state or "tfidf_texts" in st.session_state:
    st.divider()
    question = st.text_input("💬 Ask a question about the PDFs")

    if question:
        # Use ChatGroq with CORRECT MODEL NAME
        try:
            from langchain_groq import ChatGroq
            # FIXED: Using current working model
            llm = ChatGroq(
                model="llama-3.3-70b-versatile",  # ← CORRECT MODEL
                temperature=0,
                api_key=os.getenv("GROQ_API_KEY")
            )
            use_mock = False
        except Exception as e:
            st.error(f"ChatGroq failed: {e}")
            use_mock = True
        
        if use_mock:
            class MockLLM:
                def invoke(self, prompt):
                    class Response:
                        content = "Mock response - ChatGroq not available"
                    return Response()
            llm = MockLLM()

        # Retrieve relevant documents
        relevant_docs = []
        
        if "vectorstore" in st.session_state:
            # Use Chroma vector store
            docs = st.session_state.vectorstore.similarity_search(question, k=4)
            relevant_docs = docs
        else:
            # Use TF-IDF
            from sklearn.metrics.pairwise import cosine_similarity
            q_vec = st.session_state.tfidf_vectorizer.transform([question])
            similarities = cosine_similarity(q_vec, st.session_state.tfidf_matrix).flatten()
            top_indices = np.argsort(similarities)[-4:][::-1]
            
            for idx in top_indices:
                relevant_docs.append({
                    "page_content": st.session_state.tfidf_texts[idx],
                    "metadata": st.session_state.tfidf_metadatas[idx]
                })

        # Build context
        context = "\n\n".join([
            getattr(doc, 'page_content', doc.get('page_content', str(doc))) 
            for doc in relevant_docs
        ])

        prompt = f"""
Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question:
{question}
"""

        with st.spinner("Thinking..."):
            response = llm.invoke(prompt)

        st.subheader("🧠 Answer")
        answer_text = getattr(response, 'content', response)
        st.write(answer_text)

        st.subheader("📚 Sources")
        for doc in relevant_docs:
            metadata = getattr(doc, 'metadata', doc.get('metadata', {}))
            source = metadata.get('source', 'PDF')
            page = metadata.get('page', 'N/A')
            st.write(f"- {source} (page {page})")
else:
    st.info("Upload PDFs and click **Process PDFs** to begin.")