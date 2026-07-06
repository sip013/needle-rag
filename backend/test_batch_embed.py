import time
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv(dotenv_path="../.env")
import os
key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=key)

print("Testing batch embedding with gemini-embedding-2...")
start = time.time()
try:
    batch = ["Hello world 1", "Hello world 2", "Hello world 3"]
    result = client.models.embed_content(
        model="gemini-embedding-2",
        contents=batch,
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_DOCUMENT",
        ),
    )
    print("Success! Number of embeddings returned:", len(result.embeddings))
except Exception as e:
    print("Error:", e)
print("Time taken:", time.time() - start)
