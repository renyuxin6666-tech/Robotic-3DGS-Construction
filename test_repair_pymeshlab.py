import trimesh
import pymeshlab
import numpy as np
from pathlib import Path

def test_pymeshlab_repair():
    model_path = Path("stage_1/assets/branches_3d/log_model/model_1.obj")
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        return

    print(f"Testing PyMeshLab repair on {model_path}...")
    
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(str(model_path))
    
    print("Mesh loaded.")
    
    # Try different naming conventions
    try:
        print("Attempting ms.meshing_close_holes...")
        ms.meshing_close_holes(maxholesize=10000)
    except AttributeError:
        print("ms.meshing_close_holes not found. Attempting ms.close_holes...")
        try:
             ms.close_holes(maxholesize=10000)
        except AttributeError:
             print("ms.close_holes not found. Listing available methods starting with 'meshing'...")
             for method in dir(ms):
                 if 'meshing' in method or 'close' in method:
                     print(f" - {method}")
             return

    # Retrieve the mesh
    m = ms.current_mesh()
    verts = m.vertex_matrix()
    faces = m.face_matrix()
    
    # Create Trimesh object
    mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    
    print(f"Is watertight? {mesh.is_watertight}")
    boundary_edges = mesh.edges[trimesh.grouping.group_rows(mesh.edges_sorted, require_count=1)]
    print(f"Boundary edges count: {len(boundary_edges)}")

if __name__ == "__main__":
    test_pymeshlab_repair()
