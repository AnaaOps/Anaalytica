import os
import requests
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
url = "https://api.groq.com/openai/v1/chat/completions"
headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

# Test with llama-3.1-8b-instant (fast and reliable)
data = {
    "model": "llama-3.1-8b-instant",
    "messages": [{"role": "user", "content": "What is 2+2?"}],
    "temperature": 0.7,
    "max_tokens": 50
}

response = requests.post(url, headers=headers, json=data)
result = response.json()

print(f"Model: {result.get('model')}")
print(f"Status: {response.status_code}")
print(f"Response: {result['choices'][0]['message']['content']}")
