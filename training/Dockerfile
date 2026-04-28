FROM sentiment-analysis-backend:latest

USER root
WORKDIR /app

ENV PYTHONPATH=/app \
    MLFLOW_TRACKING_URI=http://mlflow:5000 \
    NLTK_DATA=/usr/local/share/nltk_data \
    HOME=/tmp

COPY backend ./backend
COPY training ./training

CMD ["python", "training/train.py"]
