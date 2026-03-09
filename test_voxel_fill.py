import trimesh
import numpy as np
from scipy import ndimage

def create_open_cylinder():
    # Create a cylinder without caps (hollow tubeish)
    # Trimesh creation usually makes watertight primitives.
    # We'll make a cylinder and delete the top/bottom faces.
    mesh = trimesh.creation.cylinder(radius=1.0, height=5.0, sections=32)
    
    # Identify faces on top and bottom caps
    # Top: z ~ 2.5, Bottom: z ~ -2.5
    centroids = mesh.triangles_center
    mask = (np.abs(centroids[:, 2]) < 2.49) # Keep only side faces
    
    mesh.update_faces(mask)
    mesh.remove_unreferenced_vertices()
    
    print(f"Created open cylinder. Watertight: {mesh.is_watertight}")
    return mesh

def fix_with_morphology(mesh, dilation_iter=2):
    print("--- Fixing with Morphological Closing (Voxelization) ---")
    
    # 1. Voxelize
    # Pitch needs to be small enough to capture detail, but large enough for dilation to close the hole
    # For a radius 1.0 cylinder, pitch 0.05 gives diameter ~40 voxels.
    pitch = 0.05
    voxel_grid = mesh.voxelized(pitch=pitch)
    
    # Get the boolean matrix
    matrix = voxel_grid.matrix
    print(f"Original Voxel Shape: {matrix.shape}")
    
    # 2. Dilate (Expand)
    # This fills the hollow inside if iterations are enough
    # If radius is 1.0 (20 voxels), we need dilation ~20 to fill it? 
    # No, we just need to close the *caps* effectively if we view it as a solid.
    # But if it's a thin shell, we want to make it solid.
    # Dilation makes the wall thicker. 
    
    # Let's use binary_closing: Dilation followed by Erosion.
    # iterations determine the size of the hole we can close.
    # To close the caps (radius 1.0), we need significant closing structure.
    
    # Actually, if we just want to cap the ends, normal fill_holes is better.
    # But if fill_holes fails, we want to force it solid.
    
    # Let's try simple dilation to thicken walls, potentially merging them?
    # No, that changes geometry too much.
    
    # Alternative: "fill_holes" on the voxel slices?
    # ndimage.binary_fill_holes fills holes inside closed regions.
    # If we slice along Z, the ring is a closed region 2D. fill_holes 2D will fill it!
    
    print("Applying slice-wise binary_fill_holes...")
    # Assuming Z is the long axis (usually true for branches?), or we try all axes.
    # For a general object, we don't know the axis.
    
    # Let's try filling holes in 3D directly? 
    # binary_fill_holes in 3D only fills voids that are completely encased. An open pipe is not encased.
    
    # Strategy: Fill holes on 2D projections (slices) along all 3 axes.
    # This is a heuristic "Orthogonal Convex Hull" intersection.
    
    matrix_filled = matrix.copy()
    
    # Fill along X axis slices
    for i in range(matrix.shape[0]):
        matrix_filled[i, :, :] = ndimage.binary_fill_holes(matrix_filled[i, :, :])
        
    # Fill along Y axis slices
    for i in range(matrix.shape[1]):
        matrix_filled[:, i, :] = ndimage.binary_fill_holes(matrix_filled[:, i, :])

    # Fill along Z axis slices
    for i in range(matrix.shape[2]):
        matrix_filled[:, :, i] = ndimage.binary_fill_holes(matrix_filled[:, :, i])
        
    # Re-create voxel grid
    # We need to use the original encoding/transform
    new_voxel = trimesh.voxel.VoxelGrid(
        trimesh.voxel.encoding.DenseEncoding(matrix_filled), 
        transform=voxel_grid.transform
    )
    
    mesh_fixed = new_voxel.marching_cubes
    print(f"Fixed Watertight: {mesh_fixed.is_watertight}")
    return mesh_fixed

if __name__ == "__main__":
    cyl = create_open_cylinder()
    fixed = fix_with_morphology(cyl)
    
    # Verify bounds
    print(f"Original Bounds: {cyl.bounds}")
    print(f"Fixed Bounds: {fixed.bounds}")
