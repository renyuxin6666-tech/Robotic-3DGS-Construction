import numpy as np

class ConfidenceScorer:
    def __init__(self, config):
        self.config = config

    def calculate(self, results, distances, selection):
        """
        Calculate confidence score.
        
        Args:
            results (list): Top-k metadata.
            distances (list/np.array): Top-k distances.
            selection (dict): Result from CoarsePoseEstimator.
            
        Returns:
            float: Confidence score [0, 1]
        """
        if not results or not selection:
            return 0.0
            
        # Metric 1: Vote Ratio (already computed in selection, but we can recompute or use it)
        vote_conf = selection['confidence']
        
        # Metric 2: Distance Gap (Margin)
        # If Top-1 is much closer than Top-2 (of a DIFFERENT branch), confidence is high.
        # If Top-1 and Top-2 are very close but different branches, confidence is low.
        
        selected_branch = selection['branch_id']
        
        # Find distance of best match for selected branch
        d1 = distances[selection['source_idx']]
        
        # Find distance of best match for ANY OTHER branch
        d2 = float('inf')
        for i, meta in enumerate(results):
            if meta['branch_id'] != selected_branch:
                d2 = distances[i]
                break
        
        margin_conf = 1.0
        if d2 != float('inf'):
            # Simple margin score: 1 - (d1 / d2)
            # If d1=0.1, d2=0.2 -> 1 - 0.5 = 0.5
            # If d1=0.1, d2=1.0 -> 1 - 0.1 = 0.9
            if d2 > 0:
                margin_conf = max(0, 1.0 - (d1 / d2))
            else:
                margin_conf = 0.0 # Should not happen if d1 < d2
        else:
            # All results are the same branch
            margin_conf = 1.0
            
        # Combine metrics (simple average or weighted)
        final_conf = 0.6 * vote_conf + 0.4 * margin_conf
        
        return min(max(final_conf, 0.0), 1.0)
