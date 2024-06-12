FROM fschlatt/natural-language-processing-exercises:0.0.1

ADD run.py /code/run.py
COPY requirements.txt /code/requirements.txt

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

ENTRYPOINT [ "python3", "/code/run.py" ]