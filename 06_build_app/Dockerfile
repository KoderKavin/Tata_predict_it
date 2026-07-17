# Predict-It — Streamlit inference UI
# Build from predict-it/ (not this folder):
#   docker build -f 06_build_app/Dockerfile -t predict-it-ui:latest .

FROM python:3.11-slim-bookworm

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# app.py resolves paths relative to predict-it/ root (parent of 06_build_app)
COPY 01_dataset/ ./01_dataset/
COPY 03_starter_kit/ ./03_starter_kit/
COPY 06_build_app/app.py ./06_build_app/

# Generate dataset and train model during build since model.joblib is ignored by Git
RUN python 03_starter_kit/train.py

WORKDIR /app/06_build_app

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
