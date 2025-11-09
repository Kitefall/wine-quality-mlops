FROM apache/airflow:3.1.0

COPY requirements.txt /requirements.txt
RUN pip install -r /requirements.txt

COPY dags/ /opt/airflow/dags/
COPY src/ /opt/airflow/src/
COPY data/ /opt/airflow/data/

WORKDIR /opt/airflow

USER root
RUN mkdir -p /opt/airflow/.dvc/tmp && \
    chown -R airflow:0 /opt/airflow/.dvc && \
    chmod -R 775 /opt/airflow/.dvc
USER airflow

CMD ["bash", "-c", "airflow db migrate && airflow api-server & airflow scheduler"]
