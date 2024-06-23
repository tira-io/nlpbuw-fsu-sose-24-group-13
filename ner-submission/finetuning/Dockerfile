# Use an appropriate base image for natural language processing exercises
FROM fschlatt/natural-language-processing-exercises:0.0.1

# Install Python 3.9 and system dependencies
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install tira rest_api_client pandas sklearn_crfsuite matplotlib

# Add the script to the image
ADD run.py /code/run.py

# Define the entry point
ENTRYPOINT ["python3", "/code/run.py"]