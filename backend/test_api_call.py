import os
import time
from google import genai
from dotenv import load_dotenv

load_dotenv(dotenv_path="../.env")
key = os.getenv("GEMINI_API_KEY")
print(f"Key loaded: {'Yes' if key else 'No'}")

client = genai.Client(api_key=key)

print("Testing text-embedding-004...")
start = time.time()
try:
    result = client.models.embed_content(
        model="text-embedding-004",
        contents="Hello world",
    )
    print("Success 004! Time:", time.time() - start)
except Exception as e:
    print("Error 004:", e)

print("Testing gemini-embedding-2...")
start = time.time()
try:
    result = client.models.embed_content(
        model="gemini-embedding-2",
        contents="Hello world",
    )
    print("Success user model! Time:", time.time() - start)
except Exception as e:
    print("Error user model:", e)
