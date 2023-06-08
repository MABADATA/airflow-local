FROM apache/airflow:2.3.0
COPY  requirements_dags.txt /requirements_dags.txt
RUN pip install --user --upgrade pip
RUN pip install --no-cache-dir --user -r /requirements_dags.txt
COPY file_loader .
