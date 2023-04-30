FROM apache/airflow:2.3.0
COPY  requirements_dags.txt /requirements_dags.txt
RUN pip install --user --upgrade pip
RUN pip install --no-cache-dir --user -r /requirements_dags.txt
ENV GOOGLE_APPLICATION_CREDENTIALS="mabadata-733abc189d01.json"
COPY dags/mabadata-733abc189d01.json .
