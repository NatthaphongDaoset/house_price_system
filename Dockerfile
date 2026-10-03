FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY serving/ ./serving/
COPY config/ ./config/
COPY artifacts/ ./artifacts/

EXPOSE 8000

CMD ["python", "-m", "serving.app"]