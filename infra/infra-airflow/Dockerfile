FROM apache/airflow:2.9.3
ARG AIRFLOW_VERSION=2.9.3
ARG PYTHON_MAJOR_MINOR=3.10
ARG AIRFLOW_CONSTRAINTS_LOCATION="https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-${PYTHON_MAJOR_MINOR}.txt"
ENV AIRFLOW_CONSTRAINTS_LOCATION=${AIRFLOW_CONSTRAINTS_LOCATION}

USER root
RUN apt-get update && \
    apt-get install -y git && \
    apt-get install -y libgomp1 && \
    apt-get install -y ffmpeg

USER airflow
# This is to fix a bug in Airflow with PostgreSQL connection
RUN pip install --no-cache-dir --constraint "${AIRFLOW_CONSTRAINTS_LOCATION}" \
    git+https://github.com/mpgreg/airflow-provider-great-expectations.git@87a42e275705d413cd4482134fc0d94fa1a68e6f

# Requirement for running Docker Operator
RUN pip install --no-cache-dir --constraint "${AIRFLOW_CONSTRAINTS_LOCATION}" \
    apache-airflow-providers-docker==3.12.2

RUN pip install --no-cache-dir --constraint "${AIRFLOW_CONSTRAINTS_LOCATION}" \
    "openai" \
    "sentence-transformers" \
    "chromadb" \
    "yt-dlp" \
    "youtube-transcript-api" \
    "requests"
