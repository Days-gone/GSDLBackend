import cv2
import torch
import fast_glcm
import numpy as np
from fastapi import FastAPI, File, UploadFile, BackgroundTasks, Form
from fastapi.responses import JSONResponse
import httpx
from io import BytesIO
from PIL import Image
import logging
import tempfile
import os
import json
from masker import CamProcessor


# 初始化FastAPI应用和日志
app = FastAPI()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 全局CamProcessor实例
processor = None


@app.on_event("startup")
async def startup_event():
    """Loads the model when the service starts.

    Raises:
        Exception: If model initialization fails.
    """
    global processor
    try:
        model_path = "./weights/camnet.pth"
        processor = CamProcessor(model_path)
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Model initialization failed: {str(e)}")
        raise


async def send_results(client_url: str, images: dict, meta_data: dict):
    """Sends processed results to the client using an asynchronous HTTP client.

    Args:
        client_url (str): The URL of the client to send results to.
        images (dict): A dictionary mapping image names to their byte data.

    Raises:
        Exception: If sending results fails due to network issues or invalid responses.
    """
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            files = []
            for img_name, img_data in images.items():
                files.append((img_name, (f"{img_name}.png", img_data, "image/png")))

            form_data = {"metadata" : json.dumps(meta_data)}
            response = await client.post(client_url, files=files, data=form_data)
            response.raise_for_status()
            logger.info(f"Results sent to {client_url} successfully")
    except Exception as e:
        logger.error(f"Failed to send results to {client_url}: {str(e)}")


def process_image_data(image_data: bytes):
    """Processes image data using the CamProcessor model.

    Args:
        image_data (bytes): The raw bytes of the input image.

    Returns:
        dict: A dictionary containing processed image data with keys 'overlay' and 'boundary',
              each mapped to their respective byte streams.

    Notes:
        The temporary file created during processing is deleted after use.
    """
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
            tmp_file.write(image_data)
            tmp_path = tmp_file.name

        # 调用模型处理
        mask, overlay, transparent, boundary, _, _ = processor.process_image(tmp_path)

        # 转换图像为字节流
        mask_bytes = cv2.imencode(".png", mask * 255)[1].tobytes()
        overlay_bytes = cv2.imencode(".png", overlay[:, :, ::-1])[1].tobytes()
        transparent_bytes = cv2.imencode(".png", transparent)[1].tobytes()
        boundary_bytes = cv2.imencode(".png", boundary)[1].tobytes()


        return {"mask" : mask_bytes,"overlay": overlay_bytes, "boundary": boundary_bytes, "transparent": transparent_bytes}
    finally:
        if tmp_path:
            os.unlink(tmp_path)


@app.post("/process")
async def process_image_endpoint(
    background_tasks: BackgroundTasks,
    image_file: UploadFile = File(...),
    metadata: str = Form(...),
):
    """Handles image processing requests and initiates background processing.

    Args:
        background_tasks (BackgroundTasks): FastAPI background tasks manager.
        image_file (UploadFile): The uploaded image file to process.
        metadata (str): A JSON string containing client_ip, pictureID, and optional callback_url.

    Returns:
        JSONResponse: A response indicating processing has started (status 202) or an error (status 500).

    Raises:
        Exception: If reading the image file, parsing metadata, or initiating processing fails.
    """
    try:
        # 解析 metadata
        metadata_dict = json.loads(metadata)
        logger.info(f"Received metadata: {metadata_dict}")

        if "client_ip" not in metadata_dict or "pictureID" not in metadata_dict:
            raise ValueError("Metadata must contain 'client_ip' and 'pictureID'")

        client_ip = metadata_dict["client_ip"]
        picture_id = metadata_dict["pictureID"]
        callback_url = metadata_dict.get("callback_url", f"http://{client_ip}:8001/call_back")  # 默认回调地址

        # 读取图像数据
        image_data = await image_file.read()
        logger.info(f"Received request from {client_ip} with pictureID: {picture_id}")

        # 添加后台任务（可以传递 picture_id 到回调）
        background_tasks.add_task(async_process_wrapper, image_data, callback_url, picture_id)

        return JSONResponse(
            status_code=202,
            content={"status": "processing_started", "client": client_ip, "pictureID": picture_id},
        )
    except ValueError as ve:
        logger.error(f"Invalid metadata: {str(ve)}")
        return JSONResponse(status_code=400, content={"error": str(ve)})
    except Exception as e:
        logger.error(f"Processing failed: {str(e)}")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/health")
async def health_response():
    """Returns the health status of the server.

    Returns:
        JSONResponse: A response indicating the server is working (status 200).
    """
    return JSONResponse(status_code=200, content={"message": "server working"})


# 更新 async_process_wrapper 以支持 picture_id
async def async_process_wrapper(image_data: bytes, callback_url: str, picture_id: str):
    """Wraps image processing and result sending in an asynchronous task."""
    try:
        processed = process_image_data(image_data)
        # 附加 picture_id 到回调数据中
        additional_data = {"pictureID": picture_id, "status": "completed"}
        await send_results(callback_url, processed, additional_data)
    except Exception as e:
        logger.error(f"Background processing failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)