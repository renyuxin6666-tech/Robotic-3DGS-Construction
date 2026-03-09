import random
import numpy as np
from PIL import Image

class RandomCropper:
    """
    负责从全局图像中生成局部裁剪样本。
    核心逻辑：
    1. 随机选择裁剪中心和尺寸
    2. 检查裁剪区域内是否有足够的前景像素（避免切到全白背景）
    3. 返回裁剪后的图像和元数据
    """
    def __init__(self, scale_range=(0.3, 0.6), min_foreground_ratio=0.1):
        self.scale_range = scale_range
        self.min_foreground_ratio = min_foreground_ratio
        
    def crop(self, image_path, num_crops=1):
        """
        对单张图片进行多次随机裁剪
        
        参数:
            image_path: 图片路径 (str)
            num_crops: 需要生成的裁剪数量 (int)
            
        返回:
            list of dict: [{"image": PIL.Image, "box": (x, y, w, h)}, ...]
        """
        # 加载图片并转为二值 mask (假设输入是黑白轮廓图)
        # 0 = 黑色 (前景), 255 = 白色 (背景)
        with Image.open(image_path) as img:
            # 转换为灰度图
            img_gray = img.convert("L")
            width, height = img.size
            
            # 将 PIL Image 转为 numpy 数组以便计算前景
            # 注意：在我们的设定中，黑色(0)是树枝，白色(255)是背景
            # 所以前景像素是 < 128 的点
            arr = np.array(img_gray)
            foreground_mask = arr < 128
            
            # 如果整张图几乎没有前景，直接跳过
            if np.sum(foreground_mask) < 100:
                return []
                
            results = []
            attempts = 0
            max_attempts = num_crops * 20 # 防止死循环
            
            while len(results) < num_crops and attempts < max_attempts:
                attempts += 1
                
                # 1. 随机生成裁剪框尺寸
                scale = random.uniform(*self.scale_range)
                crop_w = int(width * scale)
                crop_h = int(height * scale)
                
                # 2. 随机生成左上角坐标
                if width - crop_w <= 0 or height - crop_h <= 0:
                    continue
                    
                x = random.randint(0, width - crop_w)
                y = random.randint(0, height - crop_h)
                
                # 3. 检查前景比例
                # 提取裁剪区域的 mask
                crop_mask = foreground_mask[y:y+crop_h, x:x+crop_w]
                foreground_ratio = np.sum(crop_mask) / (crop_w * crop_h)
                
                if foreground_ratio >= self.min_foreground_ratio:
                    # 4. 执行裁剪
                    crop_img = img.crop((x, y, x+crop_w, y+crop_h))
                    results.append({
                        "image": crop_img,
                        "box": [x, y, crop_w, crop_h], # XYWH 格式
                        "scale": scale
                    })
                    
            return results
