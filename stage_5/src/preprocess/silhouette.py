from PIL import Image
import torchvision.transforms as T
import torch

class SilhouettePreprocessor:
    def __init__(self, image_size=224):
        self.image_size = image_size
        self.transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            # Normalize using ImageNet stats as in training
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def process(self, image_path):
        """
        Load and process an image.
        Args:
            image_path (str or Path): Path to image file.
        Returns:
            tensor: (1, 3, H, W) ready for model inference.
        """
        try:
            image = Image.open(image_path).convert("RGB")
            tensor = self.transform(image)
            return tensor.unsqueeze(0) # Add batch dimension
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            return None
