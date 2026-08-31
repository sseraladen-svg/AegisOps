FROM python:3.12-slim
WORKDIR /app

COPY requirements.txt .
# CPU-only torch first, so the requirements resolve against it instead of
# pulling the CUDA wheel and its ~2GB of runtime into the image.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch \
 && pip install --no-cache-dir -r requirements.txt

COPY . .
RUN chmod +x /app/entrypoint.sh

EXPOSE 8000
ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
