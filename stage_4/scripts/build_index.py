import sys
import os
import yaml
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader

# Add paths to sys.path to import modules from stage_3 and stage_4
# Assuming script is run from stage_4/ directory or stage_4/scripts/
current_dir = Path(__file__).resolve().parent
stage_4_root = current_dir.parent
stage_3_root = stage_4_root.parent / "stage_3"

sys.path.append(str(stage_3_root))
sys.path.append(str(stage_4_root))

from src.models.model import BranchEmbeddingModel
from src.dataset.dataset import BranchDataset
from src.indexer.faiss_engine import FaissEngine

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def build_index():
    # 1. Load Config
    config_path = stage_4_root / "config" / "index.yaml"
    config = load_config(config_path)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Prepare Data (Gallery: Global samples only)
    # We use 'test' split to build the gallery for simulation, 
    # assuming 'test' contains the inventory we want to search against.
    # In a real scenario, this would be the 'database' set.
    data_root = (stage_4_root / config['paths']['data_root']).resolve()
    dataset = BranchDataset(
        root_dir=data_root,
        split="test", # Using test set for gallery
        image_size=config['model']['image_size']
    )
    
    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4
    )
    
    # 3. Load Model
    checkpoint_path = (stage_4_root / config['paths']['model_checkpoint']).resolve()
    print(f"Loading model from {checkpoint_path}")
    
    # Force CPU for compatibility
    device = torch.device("cpu")
    print(f"Using device: {device} (Forced CPU)")

    model = BranchEmbeddingModel(
        backbone_name=config['model']['backbone'],
        embedding_dim=config['model']['embedding_dim'],
        pretrained=False # Weights loaded from checkpoint
    )
    
    # Load weights
    # The checkpoint saved in stage_3 is the state_dict directly
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    
    model.to(device)
    model.eval()
    
    # 4. Initialize Indexer
    indexer = FaissEngine(
        dim=config['index']['dim'],
        index_type=config['index']['type'],
        metric=config['index']['metric']
    )
    
    print("Start building index...")
    count = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader):
            images = batch['image'].to(device)
            branch_ids = batch['branch_id']
            is_locals = batch['is_local']
            poses = batch['pose'] # (B, 4, 4)
            
            # Filter for Global samples only (Gallery)
            # is_local is a tensor of booleans
            global_mask = ~is_locals
            
            if not global_mask.any():
                continue
                
            global_images = images[global_mask]
            global_branch_ids = np.array(branch_ids)[global_mask.cpu().numpy()]
            global_poses = poses[global_mask].cpu().numpy()
            
            # Extract embeddings
            embeddings = model(global_images)
            embeddings = embeddings.cpu().numpy()
            
            # Prepare metadata
            metas = []
            for i, bid in enumerate(global_branch_ids):
                metas.append({
                    "branch_id": bid,
                    "type": "global",
                    "pose": global_poses[i].tolist() # Save as list for JSON serialization safety if needed
                })
            
            # Add to index
            indexer.add(embeddings, metas)
            count += len(embeddings)
            
    print(f"Index built with {count} vectors.")
    
    # 5. Save Index
    index_dir = (stage_4_root / config['paths']['index_dir']).resolve()
    index_path = index_dir / config['paths']['index_file']
    meta_path = index_dir / config['paths']['meta_file']
    
    print(f"Saving index to {index_path}")
    indexer.save(index_path, meta_path)
    print("Done!")

if __name__ == "__main__":
    build_index()
