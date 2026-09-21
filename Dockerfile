FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV PYTHONUNBUFFERED=1
EXPOSE 8000
CMD ["bash", "-c", "uvicorn app.api:app --host 0.0.0.0 --port 8000 2>&1 | sed 's/^/[backend] /' & sleep 5 && streamlit run ui/streamlit_app.py --server.port ${PORT:-8501} --server.address 0.0.0.0"]
