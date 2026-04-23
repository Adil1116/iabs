FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    IABS_HOST=0.0.0.0 \
    IABS_PORT=7860 \
    IABS_LOG_LEVEL=INFO

WORKDIR /app

RUN addgroup --system iabs && adduser --system --ingroup iabs iabs

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN mkdir -p /app/data && chown -R iabs:iabs /app

USER iabs

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860/healthz', timeout=3).read()"

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]
