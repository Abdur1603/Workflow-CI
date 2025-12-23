FROM python:3.9-slim
WORKDIR /app
COPY conda.yaml .
RUN pip install pandas numpy scikit-learn mlflow matplotlib seaborn dagshub
COPY . /app
CMD ["python", "modelling.py"]