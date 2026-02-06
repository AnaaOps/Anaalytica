# rag_test.py
import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

class SimpleRAG:
    def __init__(self):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model = "llama-3.1-8b-instant"  # Fast and reliable
        self.context = ""  # This would be your vector DB retrieval in real RAG
        
    def query(self, question):
        # In real RAG, you would:
        # 1. Retrieve relevant context from vector DB
        # 2. Format prompt with context
        # 3. Send to LLM
        
        prompt = f"""
        Context: {self.context}
        
        Question: {question}
        
        Answer based on the context above:
        """
        
        response = self.client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            model=self.model,
            temperature=0.7,
            max_tokens=200
        )
        
        return response.choices[0].message.content

# Test it
rag = SimpleRAG()
print("Testing RAG setup...")
print("Response:", rag.query("What is artificial intelligence?"))