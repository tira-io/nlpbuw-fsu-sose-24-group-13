FROM fschlatt/natural-language-processing-exercises:0.0.1

ADD run.py /code/run.py
ADD utils.py /code/utils.py
COPY model.bin /code/model.bin
COPY requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

ENTRYPOINT [ "python3", "/code/run.py" ]
