from collections import Counter
import numpy as np

class CoarsePoseEstimator:
    def __init__(self, config):
        self.config = config

    def estimate(self, results, distances):
        """
        Estimate coarse pose from retrieval results.
        
        Args:
            results (list): List of metadata dicts for one query.
            distances (list or np.ndarray): Distances corresponding to results.
            
        Returns:
            dict: {
                "branch_id": str,
                "pose": list (4x4),
                "confidence": float, # Simple confidence based on vote ratio
                "source_idx": int # Index in the original results list
            }
        """
        if not results:
            return None
            
        # 1. Vote for Branch ID
        branch_ids = [meta['branch_id'] for meta in results]
        
        # Count votes
        counter = Counter(branch_ids)
        most_common = counter.most_common(1)
        
        if not most_common:
            return None
            
        best_branch_id, vote_count = most_common[0]
        
        # 2. Find the best match for this branch_id
        # We assume results are sorted by distance (ascending)
        best_match_meta = None
        best_match_idx = -1
        
        for i, meta in enumerate(results):
            if meta['branch_id'] == best_branch_id:
                best_match_meta = meta
                best_match_idx = i
                break
                
        if best_match_meta is None:
            # Should not happen logic-wise
            return None
            
        # 3. Calculate Confidence
        # Ratio of votes for the winner
        confidence = vote_count / len(results)
        
        return {
            "branch_id": best_branch_id,
            "pose": best_match_meta['pose'],
            "confidence": confidence,
            "source_idx": best_match_idx
        }
