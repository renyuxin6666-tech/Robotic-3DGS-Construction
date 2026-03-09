import sys
import os
import yaml
import torch
import numpy as np
import math
import pickle
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T

# Setup paths
current_dir = Path(__file__).resolve().parent
stage_4_new_root = current_dir.parent
project_root = stage_4_new_root.parent
stage_4_root = project_root / "stage_4"
stage_3_pkg_root = project_root / "stage_3_new"

# Add paths to sys.path
sys.path.append(str(stage_4_root))
sys.path.append(str(stage_3_pkg_root))

# Import FaissEngine from stage_4
from src.indexer.faiss_engine import FaissEngine
# Import Model from stage_3_new
from model import BranchEmbeddingModel

# --- Fibonacci Helper Functions (from stage_1/scripts/render_fibonacci.py) ---

def fibonacci_sphere(samples=1000):
    """
    Generate points on a sphere using Fibonacci Lattice.
    """
    points = []
    phi = math.pi * (3. - math.sqrt(5.))
    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2
        radius = math.sqrt(1 - y * y)
        theta = phi * i
        x = math.cos(theta) * radius
        z = math.sin(theta) * radius
        points.append([x, y, z])
    return np.array(points)

def get_look_at_matrix(eye, target, up):
    """
    Calculate Camera Pose Matrix (Camera-to-World).
    """
    z_axis = eye - target
    z_axis = z_axis / np.linalg.norm(z_axis)
    x_axis = np.cross(up, z_axis)
    if np.linalg.norm(x_axis) < 1e-6:
        x_axis = np.array([1, 0, 0])
    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)
    
    rotation = np.column_stack((x_axis, y_axis, z_axis))
    translation = eye
    
    pose = np.eye(4)
    pose[:3, :3] = rotation
    pose[:3, 3] = translation
    return pose

# --- Dataset Class ---

class FibonacciDataset(Dataset):
    def __init__(self, root_dir, image_size=224):
        self.root_dir = Path(root_dir)
        self.image_size = image_size
        self.samples = []
        
        # Scan for model folders
        model_dirs = sorted([d for d in self.root_dir.iterdir() if d.is_dir() and d.name.startswith("model_")])
        
        for model_dir in model_dirs:
            # Parse branch_id from folder name "model_1" -> 1
            try:
                branch_id = int(model_dir.name.split("_")[1])
            except:
                continue
                
            # Scan images
            images = sorted(list(model_dir.glob("*.png")))
            num_images = len(images)
            if num_images == 0:
                continue
                
            # Generate poses for this count
            # Assuming standard Fibonacci sphere centered at origin, looking at origin
            # Camera distance is usually fixed. In render_fibonacci.py it was likely 2.0 or 3.0?
            # We assume a standard distance here or unit sphere direction. 
            # For retrieval, the direction matters most.
            # Let's assume distance = 3.0 (common default) or verify from render script.
            # But the pose stored should be the camera pose.
            # Re-generating points on unit sphere:
            sphere_points = fibonacci_sphere(num_images)
            
            # Default up vector
            up = np.array([0, 1, 0])
            target = np.array([0, 0, 0])
            camera_distance = 3.0 # Assumed default, check render script if critical
            
            for i, img_path in enumerate(images):
                # Camera position
                eye = sphere_points[i] * camera_distance
                pose = get_look_at_matrix(eye, target, up)
                
                self.samples.append({
                    "image_path": str(img_path),
                    "branch_id": branch_id,
                    "view_id": i,
                    "pose": pose
                })

        self.transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        print(f"Found {len(self.samples)} images across {len(model_dirs)} models.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        image = Image.open(item["image_path"]).convert("RGB")
        if self.transform:
            image = self.transform(image)
            
        return {
            "image": image,
            "branch_id": item["branch_id"],
            "view_id": item["view_id"],
            "pose": torch.tensor(item["pose"], dtype=torch.float32),
            "image_path": item["image_path"]
        }

# --- Main Build Function ---

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def build_index():
    # 1. Load Config
    config_path = stage_4_new_root / "config" / "index.yaml"
    config = load_config(config_path)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Prepare Data
    data_root = (stage_4_new_root / config['paths']['data_root']).resolve()
    print(f"Loading data from: {data_root}")
    
    dataset = FibonacciDataset(
        root_dir=data_root,
        image_size=config['model']['image_size']
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4
    )
    
    # 3. Load Model
    checkpoint_path = (stage_4_new_root / config['paths']['model_checkpoint']).resolve()
    print(f"Loading model from {checkpoint_path}")
    
    # Initialize model
    model = BranchEmbeddingModel(
        backbone_name=config['model']['backbone'],
        embedding_dim=config['model']['embedding_dim'],
        pretrained=False
    )
    
    # Load weights
    # Note: Checkpoint might contain full state (epoch, optimizer, etc.)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        print("Loading model state from 'model_state_dict' key in checkpoint.")
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
        
    model.load_state_dict(state_dict)
    
    model.to(device)
    model.eval()
    
    # 4. Initialize Indexer
    indexer = FaissEngine(
        dim=config['index']['dim'],
        index_type=config['index']['type'],
        metric=config['index']['metric']
    )
    
    print("Start extracting embeddings and building index...")
    
    meta_data = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader):
            images = batch['image'].to(device)
            # branch_ids = batch['branch_id']
            # view_ids = batch['view_id']
            # poses = batch['pose']
            
            # Extract embeddings
            embeddings = model(images)
            embeddings_np = embeddings.cpu().numpy()
            
            # Prepare metadata for this batch
            batch_metas = []
            B = images.size(0)
            for i in range(B):
                meta = {
                    "branch_id": int(batch["branch_id"][i]),
                    "view_id": int(batch["view_id"][i]),
                    "pose": batch["pose"][i].numpy(),
                    "image_path": batch["image_path"][i]
                }
                batch_metas.append(meta)

            # Add to index (vectors + metadata)
            indexer.add(embeddings_np, batch_metas)
            
    # 5. Save Index and Metadata
    output_dir = (stage_4_new_root / config['paths']['index_dir']).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    index_path = output_dir / config['paths']['index_file']
    meta_path = output_dir / config['paths']['meta_file']
    
    print(f"Saving index and metadata to {output_dir}")
    indexer.save(index_path, meta_path)
        
    print("Index build complete!")

if __name__ == "__main__":
    build_index()
