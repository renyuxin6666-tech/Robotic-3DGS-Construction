import argparse
import sys
import yaml
import json
import torch
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw

# Add paths to ensure modules can be imported
current_dir = Path(__file__).resolve().parent
stage_5_root = current_dir.parent
stage_4_root = stage_5_root.parent / "stage_4"
stage_3_root = stage_5_root.parent / "stage_3"

sys.path.insert(0, str(stage_5_root))
sys.path.append(str(stage_3_root))
sys.path.append(str(stage_4_root))

from src.preprocess.silhouette import SilhouettePreprocessor
from src.embed.extractor import EmbeddingExtractor
from src.retrieve.search import Retriever
from src.pose.coarse import CoarsePoseEstimator
from src.score.confidence import ConfidenceScorer

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def apply_occlusion(image, ratio, mode='right_bar'):
    """
    Simulate mechanical arm occlusion.
    ratio: 0.0 to 1.0 (percentage of image width to cover)
    """
    if ratio <= 0:
        return image
    
    w, h = image.size
    draw = ImageDraw.Draw(image)
    
    # Simulate a mechanical arm entering from the right side
    # This covers 'ratio' percent of the width
    bar_width = int(w * ratio)
    
    # Draw a black rectangle (simulating the arm body)
    draw.rectangle([w - bar_width, 0, w, h], fill=(0, 0, 0))
    
    return image

def main():
    parser = argparse.ArgumentParser(description="Stage 5: Occlusion Stress Test")
    parser.add_argument("image_path", help="Path to input image")
    parser.add_argument("--config", default="configs/infer.yaml", help="Path to config file")
    args = parser.parse_args()
    
    # 1. Load Config
    config_path = stage_5_root / args.config
    if not config_path.exists():
        print(f"Config not found: {config_path}")
        return
    config = load_config(config_path)
    
    print("Initializing modules...")
    
    # 2. Init Modules
    try:
        preprocessor = SilhouettePreprocessor(image_size=config['model']['image_size'])
        extractor = EmbeddingExtractor(config)
        retriever = Retriever(config)
        estimator = CoarsePoseEstimator(config)
        scorer = ConfidenceScorer(config)
    except Exception as e:
        print(f"Module initialization failed: {e}")
        return
    
    # Load original image
    try:
        original_image = Image.open(args.image_path).convert("RGB")
    except Exception as e:
        print(f"Error loading image: {e}")
        return

    print(f"\nProcessing image: {args.image_path}")
    print(f"{'Occlusion':<10} | {'Confidence':<10} | {'Match ID':<15} | {'Status':<10}")
    print("-" * 60)

    # Baseline (0% occlusion)
    baseline_id = None
    
    # Test levels: 0% to 60% occlusion
    occlusion_levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    
    for level in occlusion_levels:
        # Create a copy to occlude
        img_copy = original_image.copy()
        occluded_img = apply_occlusion(img_copy, level, mode='right_bar')
        
        # Preprocess manually since we have a PIL image
        tensor = preprocessor.transform(occluded_img)
        image_tensor = tensor.unsqueeze(0) # Add batch dim
        
        # Pipeline
        embedding = extractor.extract(image_tensor)
        distances, results = retriever.search(embedding)
        
        # Unpack batch
        results = results[0]
        distances = distances[0]
        
        # Estimate Pose
        selection = estimator.estimate(results, distances)
        
        if selection:
            confidence = scorer.calculate(results, distances, selection)
            match_id = selection.get('branch_id', 'N/A')
            
            # Extract internal metrics for debugging
            vote_conf = selection.get('confidence', 0.0)
            top_branches = [r['branch_id'] for r in results]
            
            # Check consistency with baseline (0% occlusion)
            if level == 0.0:
                baseline_id = match_id
                status = "BASELINE"
            else:
                if match_id == baseline_id:
                    status = "PASS" if confidence > config['pipeline']['confidence_threshold'] else "LOW_CONF"
                else:
                    status = "MISMATCH"
                
            print(f"{level*100:>8.0f}% | {confidence:>10.4f} | {str(match_id):<15} | {status:<10} | Votes: {vote_conf:.2f} | Top-k: {top_branches}")
        else:
            print(f"{level*100:>8.0f}% | {'FAILED':>10} | {'N/A':<15} | {'FAIL':<10}")

if __name__ == "__main__":
    main()
