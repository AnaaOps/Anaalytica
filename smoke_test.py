import os
import tempfile
import streamlit as st
import PyPDF2
import requests
import json
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="PDF Chatbot", layout="wide")
st.title("📄 PDF Chatbot — WORKING")

# Get API key
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("""
    ❌ GROQ_API_KEY is missing!
    
    1. Create or edit .env file
    2. Add: GROQ_API_KEY=your_key_here
    3. Restart the app
    """)
    st.stop()

st.sidebar.header("⚙️ Settings")
st.sidebar.write("Current model: llama-3.3-70b-versatile")

# Upload PDFs
uploaded_files = st.file_uploader(
    "Upload PDF files",
    type="pdf",
    accept_multiple_files=True,
    help="Select one or more PDF files"
)

if st.button("🚀 Process PDFs", type="primary") and uploaded_files:
    # Read PDFs
    all_text = ""
    for pdf in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(pdf.read())
            reader = PyPDF2.PdfReader(tmp.name)
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if text.strip():
                    all_text += f"--- Page {i+1} ---\n{text}\n\n"
    
    st.session_state.pdf_text = all_text
    st.session_state.pdf_names = [pdf.name for pdf in uploaded_files]
    st.success(f"✅ Processed {len(uploaded_files)} PDF(s), {len(all_text)} characters")

# Ask questions
if "pdf_text" in st.session_state:
    st.divider()
    
    st.write(f"**Loaded PDFs:** {', '.join(st.session_state.pdf_names)}")
    
    question = st.text_input(
        "💬 Ask a question about the PDFs",
        placeholder="What would you like to know?"
    )
    
    if question:
        # Direct Groq API call - NO LANGCHAIN
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        
        # Use first 3000 characters as context
        context = st.session_state.pdf_text[:3000]
        
        data = {
            "model": "llama-3.3-70b-versatile",  # CURRENT WORKING MODEL
            "messages": [
                {
                    "role": "system",
                    "content": f"""You are a helpful assistant. Answer the question using ONLY the context below.
                    
CONTEXT FROM PDF:
{context}

INSTRUCTIONS:
1. Answer using ONLY information from the context above
2. If the answer is not in the context, say "I don't have enough information"
3. Be concise and accurate
"""
                },
                {"role": "user", "content": question}
            ],
            "temperature": 0,
            "max_tokens": 500
        }
        
        with st.spinner("🤔 Thinking..."):
            try:
                response = requests.post(url, headers=headers, json=data, timeout=30)
                
                if response.status_code == 200:
                    result = response.json()
                    answer = result["choices"][0]["message"]["content"]
                    
                    st.subheader("🧠 Answer")
                    st.write(answer)
                    
                    # Show token usage
                    with st.expander("📊 Token Usage"):
                        usage = result.get("usage", {})
                        st.write(f"Prompt tokens: {usage.get('prompt_tokens', 'N/A')}")
                        st.write(f"Completion tokens: {usage.get('completion_tokens', 'N/A')}")
                        st.write(f"Total tokens: {usage.get('total_tokens', 'N/A')}")
                        
                else:
                    error_msg = response.text
                    st.error(f"API Error {response.status_code}")
                    
                    # Check what the error is
                    if "model_decommissioned" in error_msg or "llama3-70b-8192" in error_msg:
                        st.error("""
                        ⚠️ MODEL ERROR!
                        
                        The error shows you're still trying to use the old model.
                        This means somewhere in your system, there's still a reference to 'llama3-70b-8192'.
                        
                        Solution: Restart your computer to clear all caches, then try again.
                        """)
                    else:
                        st.code(f"Error details: {error_msg[:500]}")
                        
            except requests.exceptions.Timeout:
                st.error("Request timeout. Try again.")
            except Exception as e:
                st.error(f"Error: {str(e)[:200]}")
    
    # Show context being used
    with st.expander("📄 Context being used (first 1000 chars)"):
        st.text(st.session_state.pdf_text[:1000])
        
    # Clear button
    if st.button("🗑️ Clear Session"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

else:
    # Initial instructions
    st.info("""
    ## 📋 Instructions:
    
    1. **Upload PDF files** using the uploader above
    2. Click **🚀 Process PDFs** button
    3. Ask questions about your PDFs
    
    ### Requirements:
    - `.env` file with `GROQ_API_KEY=your_key_here`
    - Python packages: `streamlit python-dotenv requests PyPDF2`
    """)

# Footer
st.sidebar.divider()
st.sidebar.caption("Made with Streamlit")