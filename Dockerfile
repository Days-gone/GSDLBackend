# 使用官方 Python 3.8 镜像作为基础镜像
FROM python:3.8-slim

# 设置工作目录
WORKDIR /app

# 安装 PDM
RUN pip install uv

# 复制项目文件（包括 pyproject.toml 和 uv.lock）
COPY pyproject.toml ./

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
# 使用 PDM 安装依赖
RUN uv sync
RUN uv add httpx python-multipart uvicorn

EXPOSE 8000

# 复制项目所有文件到容器的工作目录
COPY . .

# 设置默认的启动命令
CMD ["uv", "run","server.py"]