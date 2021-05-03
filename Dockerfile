# a Dockerfile is a build spec for a Docker image
FROM continuumio/anaconda3:2020.11

ADD . /code 
WORKDIR /code

ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
# Install dependencies:
COPY requirements.txt .
RUN pip install -r requirements.txt

ENTRYPOINT ["python", "recommendation_app.py"]