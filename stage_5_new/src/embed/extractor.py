import torch
import sys
from pathlib import Path

# Assuming sys.path includes stage_3 root
try:
    from src.models.model import BranchEmbeddingModel
except ImportError:
    try:
        from model import BranchEmbeddingModel
    except ImportError:
        # Fallback if running relative to this file (not ideal but safe)
        # This part might be tricky without correct sys.path setup from main script
        pass

class EmbeddingExtractor:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Force CPU if needed (like for RTX 5060 issue)
        self.device = torch.device("cpu") 
        print(f"Extractor using device: {self.device}")
        
        self.model = self._load_model()

    def _load_model(self):
        ckpt_path = self.config['model']['checkpoint_path']
        backbone = self.config['model']['backbone']
        dim = self.config['model']['embedding_dim']
        
        # Initialize model
        # Note: We rely on BranchEmbeddingModel being imported/available
        model = BranchEmbeddingModel(
            backbone_name=backbone,
            embedding_dim=dim,
            pretrained=False
        )
        
        # Load weights
        print(f"Loading checkpoint from {ckpt_path}")
        # Resolve path relative to stage_5 root if it's relative
        # But here we just assume the path in config is correct relative to CWD
        state_dict = torch.load(ckpt_path, map_location=self.device)
        model.load_state_dict(state_dict)
        
        model.to(self.device)
        model.eval()
        return model

    def extract(self, image_tensor):
        """
        Args:
            image_tensor: (B, 3, H, W)
        Returns:
            np.ndarray: (B, dim)
        """
        image_tensor = image_tensor.to(self.device)
        with torch.no_grad():
            embedding = self.model(image_tensor)
            return embedding.cpu().numpy()
