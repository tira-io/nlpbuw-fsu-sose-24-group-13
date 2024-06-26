# Use an appropriate base image for natural language processing exercises
FROM fschlatt/natural-language-processing-exercises:0.0.1

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install tira rest_api_client pandas sklearn_crfsuite matplotlib seqeval evaluate

# Add the script to the image
ADD run.py /code/run.py

# Define the entry point
ENTRYPOINT ["python3", "/code/run.py"]
