## 配置
本地配置
```Bash
uv sync
uv add httpx python-multipart uvicorn
apt-get install ffmpeg libsm6 libxext6  -y
```
## 运行
```Bash
uv run server.py
```