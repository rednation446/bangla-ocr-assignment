# Serving image: only the app, the trained model and what they need.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /srv/bangla-ocr

# The container only serves predictions, so it swaps TensorFlow for the smaller
# CPU build and skips the training-only packages (MLflow, matplotlib).
COPY requirements.txt .
RUN sed -e 's/^tensorflow==/tensorflow-cpu==/' -e '/^mlflow==/d' -e '/^matplotlib==/d' \
        requirements.txt > requirements-serve.txt \
 && pip install -r requirements-serve.txt

RUN useradd --create-home --uid 1001 reader
COPY --chown=reader:reader ocr_core.py app.py labels.json lexicon.txt ./
COPY --chown=reader:reader models/ ./models/
USER reader

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health')" || exit 1

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
