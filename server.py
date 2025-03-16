import cv2
import torch
import fast_glcm
import numpy as np
from fastapi import FastAPI, File, UploadFile, BackgroundTasks
from fastapi.responses import JSONResponse
import httpx
import asyncio
from io import BytesIO
from PIL import Image
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 初始化FastAPI应用
app = FastAPI(title="DL Image Processing Server")

# 设备自动检测
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")

# 加载模型
try:
    model = torch.load("./weights/camnet.pth", map_location=device).eval().to(device)
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"模型加载失败: {str(e)}")
    raise RuntimeError("Failed to initialize model")

async def process_image(image_data: bytes, callback_url: str):
    """异步处理图像的核心函数"""
    try:
        # 将字节数据转换为numpy数组
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)[:, :, ::-1]  # BGR转RGB
        
        if img is None:
            raise ValueError("无法解码图像数据")

        # 调整大小
        img = cv2.resize(img, (512, 512))
        norm_img = img / 255.0

        # 创建tensor
        tensor = torch.tensor(np.expand_dims(img, axis=0), 
                            device=device,
                            dtype=torch.float32).permute(0, 3, 1, 2)

        # 执行推理
        with torch.no_grad():
            cam = model(tensor).squeeze().cpu().numpy()

        # GLCM处理
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        glcm = fast_glcm.fast_glcm_mean(gray)
        glcm = glcm / glcm.max()

        # 融合结果
        diffused_cam = 0.5 * glcm + 0.5 * cam
        diffused_cam = np.where(diffused_cam > 0.5, 1, 0)

        # 生成结果图片
        # Mask图像
        mask_img = (diffused_cam * 255).astype(np.uint8)
        mask_buffer = BytesIO()
        Image.fromarray(mask_img).save(mask_buffer, format="JPEG")
        
        # CAM可视化
        vis_img = (norm_img * 255).astype(np.uint8)
        vis_cam = (1 - diffused_cam) * 255
        vis_img = cv2.addWeighted(vis_img, 0.7, vis_cam.astype(np.uint8), 0.3, 0)
        vis_buffer = BytesIO()
        Image.fromarray(vis_img).save(vis_buffer, format="JPEG")

        # 准备发送的回调数据
        files = {
            "mask": ("mask.jpg", mask_buffer.getvalue(), "image/jpeg"),
            "cam": ("cam.jpg", vis_buffer.getvalue(), "image/jpeg")
        }

        # 异步发送回调
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(callback_url, files=files)
            response.raise_for_status()
            logger.info(f"Callback to {callback_url} successful")

    except Exception as e:
        logger.error(f"处理失败: {str(e)}")
        # 可以添加重试逻辑或错误通知机制

@app.post("/process")
async def process_endpoint(
    background_tasks: BackgroundTasks,
    image: UploadFile = File(...),
    callback_url: str = "http://localhost:8000/receive"  # 默认值用于测试
):
    """处理端点接收图像和回调URL"""
    try:
        # 读取图像数据
        image_data = await image.read()
        
        # 添加后台任务
        background_tasks.add_task(process_image, image_data, callback_url)
        
        return JSONResponse(
            status_code=202,
            content={
                "message": "任务已接收，正在处理",
                "callback_url": callback_url
            }
        )
    
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"message": f"服务器错误: {str(e)}"}
        )

# 测试用接收端点（可选）
@app.post("/receive")
async def test_receive_endpoint(mask: UploadFile = File(...), cam: UploadFile = File(...)):
    """用于测试的接收端点"""
    return {
        "mask_size": len(await mask.read()),
        "cam_size": len(await cam.read())
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
