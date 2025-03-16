import cv2
import torch
import fast_glcm
import numpy as np
from tqdm import tqdm
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
import os

# 设备自动检测
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 输入输出路径
input_folder = r"./image"
output_folder = r"./mask"

# 加载模型（添加map_location参数）
try:
    model = torch.load("./weights/camnet.pth", map_location=device).eval().to(device)
except Exception as e:
    raise RuntimeError(f"模型加载失败: {str(e)}")

# 图像扩展名列表
IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']

# 创建输出目录
os.makedirs(output_folder, exist_ok=True)

for filename in tqdm(os.listdir(input_folder), desc="Processing images"):
    # 跳过非图像文件
    if not any(filename.lower().endswith(ext) for ext in IMAGE_EXTENSIONS):
        continue

    try:
        # 读取并预处理图像
        img_path = os.path.join(input_folder, filename)
        img = cv2.imread(img_path)[:, :, ::-1]  # BGR转RGB
        if img is None:
            raise ValueError(f"无法读取图像文件: {filename}")
            
        img = cv2.resize(img, (512, 512))
        norm_img = img / 255.0

        # 创建tensor并移动到对应设备
        tensor = torch.tensor(np.expand_dims(img, axis=0), 
                            device=device,  # 直接创建在目标设备
                            dtype=torch.float32).permute(0, 3, 1, 2)

        # 执行推理
        with torch.no_grad():
            cam = model(tensor).squeeze().cpu().numpy()

        # GLCM处理
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        glcm = fast_glcm.fast_glcm_mean(gray)
        glcm = glcm / glcm.max()

        # 融合CAM和GLCM
        diffused_cam = 0.5 * glcm + 0.5 * cam
        diffused_cam = np.where(diffused_cam > 0.5, 1, 0)  # 二值化

        # 保存结果
        base_name = os.path.splitext(filename)[0]
        
        # 保存mask
        mask_img = (diffused_cam * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(output_folder, f"{base_name}_mask.jpg"), mask_img)
        
        # 保存可视化结果
        vis_img = show_cam_on_image(norm_img, 1 - diffused_cam, use_rgb=True)
        cv2.imwrite(os.path.join(output_folder, f"{base_name}_cam.jpg"), vis_img[:, :, ::-1])  # RGB转BGR保存

    except Exception as e:
        print(f"处理 {filename} 时发生错误: {str(e)}")
        continue

print("----------------- ALL FINSHED! ---------------------")
