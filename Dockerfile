FROM python:3.8-slim

WORKDIR /app

RUN pip install uv

COPY pyproject.toml ./

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN uv sync
RUN uv add httpx python-multipart uvicorn

EXPOSE 8000

COPY . .
RUN chmod +x work.sh

CMD ["./work.sh"]