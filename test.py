import cv2
import torch
from tqdm import tqdm
from masker import CamProcessor
import os

device = torch.device('cpu')
print(f"Using device: {device}")

input_folder = r"./image"
output_folder = r"./mask"
IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']


os.makedirs(output_folder, exist_ok=True)
for filename in tqdm(os.listdir(input_folder), desc="Processing images"):
    if not any(filename.lower().endswith(ext) for ext in IMAGE_EXTENSIONS):
        continue
    try:
        img_path = os.path.join(input_folder, filename)
        processor = CamProcessor("./weights/camnet.pth")
        mask, overlay, transparent, boundary, entropy, mr = processor.process_image(img_path)

        # mask: ndarry(原图h，原图w)
        # overlay: ndarry(原图h，原图w, 3)
        # transparent: ndarry(原图h，原图w, 4)
        # boundary: ndarry(原图h，原图w, 3)
        # entropy, mr: float32, 64
        base_name = os.path.splitext(filename)[0]
        cv2.imwrite(os.path.join(output_folder, f"{base_name}_mask.png"), mask * 255)
        cv2.imwrite(os.path.join(output_folder, f"{base_name}_overlay.png"), overlay[:, :, ::-1])
        cv2.imwrite(os.path.join(output_folder, f"{base_name}_transparent.png"), transparent)
        cv2.imwrite(os.path.join(output_folder, f"{base_name}_boundary.png"), boundary)
        print("矿层复合率:", entropy)
        print("岩层丰富度:", mr)

    except Exception as e:
        print(f"处理 {filename} 时发生错误: {str(e)}")
        continue

print("----------------- ALL FINISHED! ---------------------")