FROM python:3.9-slim
WORKDIR /app
RUN apt-get update && apt-get install -y libgomp1
RUN pip install pandas numpy scikit-learn mlflow matplotlib seaborn dagshub
COPY . /app
CMD ["python", "modelling.py"]