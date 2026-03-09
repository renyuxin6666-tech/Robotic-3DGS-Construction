import argparse
import sys
import yaml
import json
import torch
import numpy as np
from pathlib import Path

# Add paths
current_dir = Path(__file__).resolve().parent
stage_5_root = current_dir.parent
stage_4_root = stage_5_root.parent / "stage_4"
stage_3_root = stage_5_root.parent / "stage_3_new"

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

def main():
    parser = argparse.ArgumentParser(description="Stage 5: Online Inference & Pose Estimation")
    parser.add_argument("image_path", help="Path to input image")
    parser.add_argument("--config", default="configs/infer.yaml", help="Path to config file")
    args = parser.parse_args()
    
    # 1. Load Config
    config_path = stage_5_root / args.config
    config = load_config(config_path)
    
    print("Initializing modules...")
    
    # 2. Init Modules
    preprocessor = SilhouettePreprocessor(image_size=config['model']['image_size'])
    extractor = EmbeddingExtractor(config)
    retriever = Retriever(config)
    estimator = CoarsePoseEstimator(config)
    scorer = ConfidenceScorer(config)
    
    # 3. Process Input
    print(f"Processing image: {args.image_path}")
    image_tensor = preprocessor.process(args.image_path)
    
    if image_tensor is None:
        print("Failed to process image.")
        return
        
    # 4. Pipeline
    
    # Step 1: Embed
    print("Extracting embedding...")
    embedding = extractor.extract(image_tensor)
    
    # Step 2: Retrieve
    print("Searching in index...")
    distances, results = retriever.search(embedding)
    
    # Flatten results (since batch size is 1)
    results = results[0]
    distances = distances[0]
    
    # Step 3: Estimate Pose
    print("Estimating pose...")
    selection = estimator.estimate(results, distances)
    
    if selection is None:
        print("No match found.")
        return
        
    # Step 4: Score Confidence
    confidence = scorer.calculate(results, distances, selection)
    selection['final_confidence'] = confidence
    
    # 5. Output
    print("\n" + "="*50)
    print("Inference Result")
    print("="*50)
    print(json.dumps(selection, indent=2, default=str))
    print("="*50)
    
    # Threshold check
    threshold = config['pipeline']['confidence_threshold']
    if confidence >= threshold:
        print("✅ High confidence match.")
    else:
        print("⚠️ Low confidence match.")

if __name__ == "__main__":
    main()
