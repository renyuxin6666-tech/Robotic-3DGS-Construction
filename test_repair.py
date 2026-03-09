import trimesh
import numpy as np
from pathlib import Path

def test_repair():
    # Path to a model
    model_path = Path("stage_1/assets/branches_3d/log_model/model_1.obj")
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        return

    print(f"Loading {model_path}...")
    mesh = trimesh.load(model_path, force='mesh')
    
    print(f"Original Is watertight: {mesh.is_watertight}")
    print(f"Original Euler number: {mesh.euler_number}")
    
    # 1. Merge vertices
    mesh.merge_vertices()
    print("Merged vertices.")
    
    # 2. Remove unreferenced
    mesh.remove_unreferenced_vertices()
    
    # Check for open edges (boundaries)
    # edges_unique are edges that only appear once (boundary)
    # But trimesh.grouping.group_rows(mesh.edges_sorted, require_count=1) gives boundary edges
    boundary_edges = mesh.edges[trimesh.grouping.group_rows(mesh.edges_sorted, require_count=1)]
    print(f"Boundary edges count: {len(boundary_edges)}")
    
    # 3. Fill holes
    print("Attempting to fill holes...")
    trimesh.repair.fill_holes(mesh)
    
    print(f"After fill_holes Is watertight: {mesh.is_watertight}")
    
    boundary_edges_after = mesh.edges[trimesh.grouping.group_rows(mesh.edges_sorted, require_count=1)]
    print(f"Boundary edges count after: {len(boundary_edges_after)}")
    
    if not mesh.is_watertight:
        print("Still not watertight. Trying more aggressive fix...")
        # Sometimes 'broken_faces' removal helps before filling?
        trimesh.repair.fix_winding(mesh)
        trimesh.repair.fix_inversion(mesh)
        trimesh.repair.fix_normals(mesh)
        
        # Try filling again
        trimesh.repair.fill_holes(mesh)
        print(f"Final Is watertight: {mesh.is_watertight}")

if __name__ == "__main__":
    test_repair()
