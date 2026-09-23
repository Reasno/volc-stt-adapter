FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt ./
# ONNX-only installation works on Python 3.11/aarch64 and avoids the missing
# tflite-runtime wheel pulled by openwakeword package metadata.
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir --no-deps openwakeword==0.6.0

COPY volc_stt_adapter.py audio_gate.py wake_word.py ./
COPY models/ ./models/

EXPOSE 8765

CMD ["python", "volc_stt_adapter.py"]
