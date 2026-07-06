import os
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()
client = genai.Client()

print("Testing text-embedding-004...")
try:
    result = client.models.embed_content(
        model="text-embedding-004",
        contents="Hello world",
    )
    print("Success 004! Embedding length:", len(result.embeddings[0].values))
except Exception as e:
    print("Error 004:", e)

print("Testing gemini-embedding-exp-03-07...")
try:
    result = client.models.embed_content(
        model="gemini-embedding-exp-03-07",
        contents="Hello world",
    )
    print("Success exp! Embedding length:", len(result.embeddings[0].values))
except Exception as e:
    print("Error exp:", e)
