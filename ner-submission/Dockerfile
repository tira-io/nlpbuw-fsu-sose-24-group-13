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
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Download NLTK data
RUN python -m nltk.downloader -d /usr/local/share/nltk_data conll2002

# Create a workspace directory
WORKDIR /workspace

# Add the source code
ADD . /workspace

# Set the default command
CMD ["bash"]
