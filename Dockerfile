# Use an official lightweight Python image
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Install system dependencies (needed for some Python packages)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create the chainlit local data directory to prevent permission errors
RUN mkdir -p /.chainlit && chmod 777 /.chainlit

# Copy the rest of the application code
COPY . .

# Expose port 7860 for Hugging Face Spaces
EXPOSE 7860

# Set environment variables for the port and python unbuffered mode
ENV PORT=7860
ENV PYTHONUNBUFFERED=1

# Command to run the application via Uvicorn (which now hosts both FastAPI and Chainlit)
CMD ["sh", "-c", "python init_chainlit_db.py && uvicorn backend.main:app --host 0.0.0.0 --port 7860"]
