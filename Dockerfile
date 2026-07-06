# Use the official Python 3.13 image
FROM python:3.13-slim

# Set the working directory
WORKDIR /app

# Install basic system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy the backend requirements file and install dependencies
COPY backend/requirements.txt ./backend/
RUN pip install --no-cache-dir -r backend/requirements.txt

# Copy the entire project into the container
COPY . .

# Expose port 7860 for Hugging Face Spaces
EXPOSE 7860

# Run the FastAPI server on port 7860
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "7860"]
