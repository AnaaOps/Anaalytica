import os
import json
import requests
from dotenv import load_dotenv

load_dotenv()

print("Testing Groq API directly...")
print("=" * 50)

api_key = os.getenv("GROQ_API_KEY")
print(f"API Key exists: {bool(api_key)}")

if not api_key:
    print("❌ ERROR: No API key found")
    exit(1)

url = "https://api.groq.com/openai/v1/chat/completions"
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

# Test 1: llama-3.3-70b-versatile
data = {
    "model": "llama-3.3-70b-versatile",
    "messages": [{"role": "user", "content": "Say hello"}],
    "temperature": 0.7,
    "max_tokens": 50
}

print(f"\n1. Testing model: {data['model']}")
response = requests.post(url, headers=headers, json=data)
print(f"Status: {response.status_code}")

if response.status_code == 200:
    result = response.json()
    print(f"✅ SUCCESS!")
    print(f"Model: {result.get('model')}")
    print(f"Response ID: {result.get('id')}")
    message_content = result['choices'][0]['message']['content']
    print(f"Message: '{message_content}'")
else:
    print(f"❌ FAILED!")
    print(f"Error: {response.text}")

# Test 2: Alternative model
print(f"\n" + "="*50)
print(f"2. Testing alternative model: mixtral-8x7b-32768")
data["model"] = "mixtral-8x7b-32768"

response = requests.post(url, headers=headers, json=data)
print(f"Status: {response.status_code}")

if response.status_code == 200:
    result = response.json()
    print(f"✅ SUCCESS!")
    print(f"Model: {result.get('model')}")
    message_content = result['choices'][0]['message']['content']
    print(f"Message: '{message_content}'")
else:
    print(f"❌ FAILED!")
    print(f"Error: {response.text}")
