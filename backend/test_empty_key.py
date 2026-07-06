import time
from google import genai
from google.genai import types

print("Testing with empty API key...")
try:
    client = genai.Client(api_key="")
    start = time.time()
    result = client.models.embed_content(
        model="text-embedding-004",
        contents="Hello world",
    )
    print("Success! Embedding length:", len(result.embeddings[0].values))
except Exception as e:
    print("Error:", e)
print(f"Time taken: {time.time() - start:.2f} seconds")
