import cv2
import torch
import fast_glcm
import numpy as np
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image


class CamProcessor:
    def __init__(self, model_path):
        self.model = torch.load(model_path, map_location='cpu').eval()
        self.image_size = (512, 512)

    def process_image(self, image_path):
        img = self._load_and_preprocess(image_path)
        cam = self._generate_cam(img)
        glcm = self._generate_glcm(img)
        diffused_cam = self._fuse_features(cam, glcm)
        binary_mask = diffused_cam.astype(np.uint8)
        overlay_img = self._create_overlay(img, diffused_cam)
        transparent_img = self._create_transparent(img, diffused_cam)

        return binary_mask, overlay_img, transparent_img

    def _load_and_preprocess(self, image_path):
        img = cv2.imread(image_path)[:, :, ::-1]
        img = cv2.resize(img, self.image_size)
        return img

    def _generate_cam(self, img):
        tensor = preprocess_image(img, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        tensor = tensor.to('cpu')

        with torch.no_grad():
            cam = self.model(tensor).squeeze().numpy()
        return (cam - cam.min()) / (cam.max() - cam.min())

    def _generate_glcm(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        glcm = fast_glcm.fast_glcm_mean(gray)
        return glcm / glcm.max()

    def _fuse_features(self, cam, glcm):
        diffused = 0.5 * glcm + 0.5 * cam
        return np.where(diffused > 0.5, 1, 0).astype(np.float32)

    def _create_overlay(self, img, mask):
        norm_img = img / 255.0
        return show_cam_on_image(norm_img, mask, use_rgb=True)

    def _create_transparent(self, img, mask):
        rgba = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)
        rgba[:, :, 3] = (mask * 255).astype(np.uint8)
        return rgba


# 使用示例
if __name__ == "__main__":
    processor = CamProcessor("./weights/camnet.pth")

    mask, overlay, transparent = processor.process_image("./image/test2.jpg")

    cv2.imwrite("./mask/mask.png", mask * 255)
    cv2.imwrite("./mask/overlay.png", overlay[:, :, ::-1])
    cv2.imwrite("./mask/transparent.png", transparent)
