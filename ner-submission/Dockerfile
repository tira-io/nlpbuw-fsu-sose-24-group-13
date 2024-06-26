# Dockerfile
FROM fschlatt/natural-language-processing-exercises:0.0.1

RUN pip install sklearn-crfsuite

ADD run.py /code/run.py
ADD train.py /code/train.py
ADD model.joblib /code/model.joblib

ENTRYPOINT ["python3", "/code/run.py"]
