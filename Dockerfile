FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000 8501
# Backend on 8000, UI on 8501 (UI reads API_URL, default http://127.0.0.1:8000)
CMD ["bash", "-c", "uvicorn app.api:app --host 0.0.0.0 --port 8000 & streamlit run ui/streamlit_app.py --server.port 8501 --server.address 0.0.0.0"]
