import trimesh
import pymeshlab
import numpy as np
from pathlib import Path

def analyze_and_fix(model_path_str):
    model_path = Path(model_path_str)
    if not model_path.exists():
        print(f"File not found: {model_path}")
        return

    print(f"Analyzing {model_path.name}...")
    
    # 1. Load Original with Trimesh
    mesh_orig = trimesh.load(model_path, force='mesh')
    print(f"[Original] Watertight: {mesh_orig.is_watertight}")
    print(f"[Original] Euler Number: {mesh_orig.euler_number}")
    boundary_edges = mesh_orig.edges[trimesh.grouping.group_rows(mesh_orig.edges_sorted, require_count=1)]
    print(f"[Original] Open edges (boundaries): {len(boundary_edges)}")

    # 2. PyMeshLab Fix (Current approach)
    print("\n--- Applying PyMeshLab Fix ---")
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(str(model_path))
    
    try:
        ms.meshing_merge_close_vertices(threshold=pymeshlab.PercentageValue(0.01))
        ms.meshing_repair_non_manifold_edges(method='Remove Faces')
        ms.meshing_repair_non_manifold_vertices()
        # Increasing maxholesize significantly
        ms.meshing_close_holes(maxholesize=1000000) 
        ms.meshing_re_orient_faces_coherently()
        
        # Export to temp to reload with trimesh
        ms.save_current_mesh("temp_pymeshlab_fix.obj")
        mesh_fix1 = trimesh.load("temp_pymeshlab_fix.obj", force='mesh')
        
        print(f"[PyMeshLab] Watertight: {mesh_fix1.is_watertight}")
        boundary_edges_1 = mesh_fix1.edges[trimesh.grouping.group_rows(mesh_fix1.edges_sorted, require_count=1)]
        print(f"[PyMeshLab] Open edges: {len(boundary_edges_1)}")
        
    except Exception as e:
        print(f"PyMeshLab failed: {e}")

    # 3. Trimesh Voxelization (The "Nuclear" Option)
    print("\n--- Applying Trimesh Voxelization ---")
    try:
        # Pitch determines resolution. Smaller is finer.
        # For a branch, we need enough resolution.
        # Scale is roughly bounding box size.
        scale = mesh_orig.scale
        pitch = scale / 100.0 # 100 voxels across
        print(f"Voxelizing with pitch {pitch}...")
        
        voxelized = mesh_orig.voxelized(pitch=pitch)
        mesh_fix2 = voxelized.marching_cubes
        
        print(f"[Voxel] Watertight: {mesh_fix2.is_watertight}")
        print(f"[Voxel] Faces: {len(mesh_fix2.faces)}")
        
    except Exception as e:
        print(f"Voxelization failed: {e}")

if __name__ == "__main__":
    # Test on one model
    target_model = "stage_1/assets/branches_3d/log_model/model_1.obj"
    analyze_and_fix(target_model)
