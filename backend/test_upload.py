import os
from rag_engine import process_document

if __name__ == "__main__":
    with open("test.txt", "w") as f:
        f.write("This is a test document. " * 100)

    with open("test.txt", "rb") as f:
        file_bytes = f.read()

    print("Starting process_document...")
    try:
        info = process_document(file_bytes, "test.txt", "text/plain")
        print("Success:", info)
    except Exception as e:
        print("Error:", e)
