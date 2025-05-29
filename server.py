import cv2
import torch
import fast_glcm
import numpy as np
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse, StreamingResponse
import httpx
from io import BytesIO
from PIL import Image
import logging
import tempfile
import os
import json
from masker import CamProcessor

# Initialize FastAPI app and logging
app = FastAPI()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global CamProcessor instance
processor = None

@app.on_event("startup")
async def startup_event():
    """Loads the model when the service starts."""
    global processor
    try:
        model_path = "./weights/camnet.pth"
        processor = CamProcessor(model_path)
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Model initialization failed: {str(e)}")
        raise

def process_image_data(image_data: bytes):
    """Processes image data and returns processed images as bytes."""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
            tmp_file.write(image_data)
            tmp_path = tmp_file.name

        mask, overlay, transparent, boundary, _, _ = processor.process_image(tmp_path)

        mask_bytes = cv2.imencode(".png", mask * 255)[1].tobytes()
        overlay_bytes = cv2.imencode(".png", overlay[:, :, ::-1])[1].tobytes()
        transparent_bytes = cv2.imencode(".png", transparent)[1].tobytes()
        boundary_bytes = cv2.imencode(".png", boundary)[1].tobytes()

        return {
            "mask": ("mask.png", mask_bytes, "image/png"),
            "overlay": ("overlay.png", overlay_bytes, "image/png"),
            "transparent": ("transparent.png", transparent_bytes, "image/png"),
            "boundary": ("boundary.png", boundary_bytes, "image/png"),
        }
    finally:
        if 'tmp_path' in locals() and tmp_path:
            os.unlink(tmp_path)

@app.post("/process_image")
async def process_image(
    image: UploadFile = File(...),
    uuid: str = Form(...)
):
    """Processes the uploaded image and returns the four processed images as multipart/form-data."""
    try:
        image_data = await image.read()
        processed_images = process_image_data(image_data)

        def generate():
            boundary = "boundary"
            for name, (filename, content, content_type) in processed_images.items():
                yield (f'--{boundary}\r\n'
                       f'Content-Disposition: form-data; name="{uuid}_{name}.png"; filename="{uuid}_{name}.png"\r\n'
                       f'Content-Type: {content_type}\r\n'
                       f'\r\n').encode('utf-8')
                yield content
                yield b'\r\n'
            yield (f'--{boundary}--\r\n').encode('utf-8')

        media_type = f"multipart/form-data; boundary=boundary"
        return StreamingResponse(content=generate(), media_type=media_type)

    except Exception as e:
        logger.error(f"Error processing image: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/health")
async def health_response():
    """Returns the health status of the server."""
    return JSONResponse(status_code=200, content={"message": "server working"})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)