import os
import time
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
client = genai.Client()

print("Testing gemini-embedding-2...")
start = time.time()
try:
    result = client.models.embed_content(
        model="gemini-embedding-2",
        contents="Hello world",
    )
    print("Success! Embedding length:", len(result.embeddings[0].values))
except Exception as e:
    print("Error:", e)
print(f"Time taken: {time.time() - start:.2f} seconds")
