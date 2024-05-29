FROM fschlatt/natural-language-processing-exercises:0.0.1

ADD run.py /code/run.py
ADD utils.py /code/utils.py
COPY model.bin /code/model.bin
COPY requirements.txt /code/requirements.txt

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -r /code/requirements.txt

ENTRYPOINT [ "python3", "/code/run.py" ]
