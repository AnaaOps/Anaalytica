import os
import sys
from dotenv import load_dotenv

print("DIAGNOSING GROQ API ISSUE")
print("=" * 60)

# Load environment variables
load_dotenv()

# Get the API key
GROQ_API_KEY = os.getenv('GROQ_API_KEY')

if not GROQ_API_KEY:
    print("X ERROR: GROQ_API_KEY not found in .env file")
    print("Add this to your .env file:")
    print("GROQ_API_KEY=your_actual_key_here")
    exit(1)

print(f"Found API key: {GROQ_API_KEY[:10]}...")  # Show first 10 chars only for security

# Test 1: Check if key format looks right
print("\n1. Checking API key format...")
if not GROQ_API_KEY.startswith('gsk_'):
    print(f"WARNING: API key doesn't start with 'gsk_' (starts with: {GROQ_API_KEY[:4]})")
else:
    print("✓ API key format looks correct")

# Test 2: Check .env file exists
print("\n2. Checking .env file...")
if os.path.exists('.env'):
    print("✓ .env file exists")
    # Show first few lines of .env (masking sensitive data)
    with open('.env', 'r') as f:
        content = f.read()
        lines = content.split('\n')
        masked_lines = []
        for line in lines:
            if 'API' in line.upper() or 'KEY' in line.upper() or 'SECRET' in line.upper():
                if '=' in line:
                    key, value = line.split('=', 1)
                    masked_lines.append(f"{key}=[MASKED]")
                else:
                    masked_lines.append("[MASKED]")
            else:
                masked_lines.append(line)
        print(f"Contents (masked):\n" + "\n".join(masked_lines[:5]))
else:
    print("X .env file not found")

print("\n3. Checking Python environment...")
print(f"Python version: {sys.version}")
print(f"Current directory: {os.getcwd()}")

print("\n" + "=" * 60)
print("Diagnosis complete.")