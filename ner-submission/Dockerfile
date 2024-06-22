# docker build -t fschlatt/authorship-verification-trivial:0.0.1 .
FROM fschlatt/natural-language-processing-exercises:0.0.1

# Use an appropriate base image
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install tira rest_api_client pandas sklearn_crfsuite matplotlib nltk

# Add the script to the image
ADD run.py /code/run.py

# Download NLTK data
RUN python -m nltk.downloader conll2002

# Define the entry point
ENTRYPOINT ["python3", "/code/run.py"]
