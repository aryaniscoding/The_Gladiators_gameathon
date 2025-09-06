FROM python:3.10-slim
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY run_model.py .
COPY ipl_2025_gameathon.xlsx /app/data/ipl_2025_gameathon.xlsx



ENTRYPOINT ["python", "run_model.py"]