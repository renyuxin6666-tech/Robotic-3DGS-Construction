import argparse
import os
import cv2
import numpy as np
import torch
from pathlib import Path
from ultralytics import YOLO
from PIL import Image

def load_clip_model():
    """
    Load CLIP model from transformers library.
    We use a small but effective CLIP model: openai/clip-vit-base-patch32
    """
    try:
        from transformers import CLIPProcessor, CLIPModel
        print("Loading CLIP model (openai/clip-vit-base-patch32)...")
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        return model, processor
    except ImportError:
        print("Error: transformers library not found. Please install it: pip install transformers")
        return None, None
    except Exception as e:
        print(f"Error loading CLIP: {e}")
        return None, None

def crop_image_by_mask(image_path, mask):
    """
    Crop the image based on the mask bounding box.
    Returns PIL Image.
    """
    # Load original image (BGR -> RGB)
    img = cv2.imread(str(image_path))
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Find contours to get bounding box
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
        
    # Get bounding box of the largest contour (assuming main object)
    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    
    # Crop
    crop = img[y:y+h, x:x+w]
    
    # Apply mask to the crop (black out background)
    mask_crop = mask[y:y+h, x:x+w]
    crop_masked = cv2.bitwise_and(crop, crop, mask=mask_crop)
    
    # Convert to PIL
    return Image.fromarray(crop_masked)

def main():
    parser = argparse.ArgumentParser(description="Evaluate YOLOv8-seg/SAM on onsite pictures with CLIP filtering")
    parser.add_argument("--input_dir", default="Onsite_picture", help="Directory containing onsite images")
    parser.add_argument("--output_dir", default="Onsite_picture/results_sam_clip", help="Directory to save results")
    parser.add_argument("--model", default="models/mobile_sam.pt", help="Model path (default: models/mobile_sam.pt)")
    parser.add_argument("--prompts", default="a tree branch, wood log, timber", help="Comma separated positive prompts")
    args = parser.parse_args()

    input_path = Path(args.input_dir)
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Check if input directory exists
    if not input_path.exists():
        print(f"Error: Input directory '{input_path}' does not exist.")
        return

    # 1. Load SAM
    print(f"Loading SAM model: {args.model}")
    try:
        from ultralytics import SAM
        sam_model = SAM(args.model)
    except Exception as e:
        print(f"Error loading SAM: {e}")
        return

    # 2. Load CLIP
    clip_model, clip_processor = load_clip_model()
    if clip_model is None:
        return
        
    # Prepare text prompts
    positive_prompts = [p.strip() for p in args.prompts.split(",")]
    # Add some negative prompts to help differentiation
    negative_prompts = ["a person", "a hand", "a robot arm", "a table", "background", "wall", "metal"]
    all_prompts = positive_prompts + negative_prompts
    print(f"CLIP Prompts: {all_prompts}")

    image_files = [f for f in input_path.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}]

    if not image_files:
        print(f"No images found in {input_path}")
        return

    print(f"Found {len(image_files)} images. Starting processing...")

    for img_file in image_files:
        print(f"\nProcessing: {img_file.name}")
        
        # A. Run SAM
        # Force CPU execution for stability
        results = sam_model(str(img_file), verbose=False, device='cpu')
        
        best_score = -1.0
        best_mask = None
        best_label = "unknown"
        
        result = results[0] # Single image inference
        
        if result.masks is None:
            print("  No masks detected by SAM.")
            continue
            
        masks = result.masks.data.cpu().numpy() # (N, H, W)
        print(f"  SAM generated {len(masks)} candidate masks.")
        
        # B. Iterate over masks and score with CLIP
        for i, mask in enumerate(masks):
            # Preprocess mask
            mask = mask.astype(np.uint8)
            mask_resized = cv2.resize(mask, (result.orig_shape[1], result.orig_shape[0]), interpolation=cv2.INTER_NEAREST)
            
            # Crop image content
            crop_pil = crop_image_by_mask(img_file, mask_resized)
            if crop_pil is None or crop_pil.width < 10 or crop_pil.height < 10:
                continue # Skip tiny masks
                
            # CLIP Inference
            try:
                inputs = clip_processor(text=all_prompts, images=crop_pil, return_tensors="pt", padding=True)
                with torch.no_grad():
                    outputs = clip_model(**inputs)
                
                # Get probabilities
                logits_per_image = outputs.logits_per_image # this is the image-text similarity score
                probs = logits_per_image.softmax(dim=1) # softmax to get probabilities
                
                # Sum probabilities of all positive prompts
                # positive prompts are the first len(positive_prompts) elements
                pos_score = probs[0, :len(positive_prompts)].sum().item()
                
                # Check if this is the best mask so far
                if pos_score > best_score:
                    best_score = pos_score
                    best_mask = mask_resized
                    # Find which specific prompt triggered it (for debugging)
                    top_idx = probs[0].argmax().item()
                    best_label = all_prompts[top_idx]
                    
            except Exception as e:
                print(f"  CLIP error on mask {i}: {e}")
                continue

        # C. Save the best result
        if best_mask is not None and best_score > 0.5: # Threshold can be tuned
            print(f"  ✅ Found best match! Label: '{best_label}' (Score: {best_score:.4f})")
            
            # Save the binary mask
            mask_out_path = output_path / f"mask_best_{img_file.stem}.png"
            cv2.imwrite(str(mask_out_path), best_mask * 255)
            
            # Save a visualization (Overlay)
            img_bgr = cv2.imread(str(img_file))
            # Green overlay
            overlay = np.zeros_like(img_bgr)
            overlay[best_mask > 0] = [0, 255, 0] 
            # Blend
            vis = cv2.addWeighted(img_bgr, 0.7, overlay, 0.3, 0)
            # Draw text
            cv2.putText(vis, f"{best_label}: {best_score:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            vis_out_path = output_path / f"vis_best_{img_file.stem}.jpg"
            cv2.imwrite(str(vis_out_path), vis)
        else:
            print(f"  ⚠️ No confident match found. Best score: {best_score:.4f} ({best_label})")

    print(f"\nProcessing complete. Results saved to: {output_path.absolute()}")

if __name__ == "__main__":
    main()
