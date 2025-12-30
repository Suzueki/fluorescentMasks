import os
import nibabel as nib
import numpy as np
from scipy.ndimage import gaussian_filter, rotate

# Configuration
FROM_PATH = r"C:\Users\marut\OneDrive\Desktop\..UofT\'projects'\Masks\Data\Masks-Pretreat"
TO_PATH = r"C:\Users\marut\OneDrive\Desktop\..UofT\'projects'\Masks\SyntheticData"
TRAINING_PATH = r"C:\Users\marut\OneDrive\Desktop\..UofT\'projects'\Masks\SyntheticDataViews"

def generate_3d_fluorescence_map(seg_data, t1_data):
    """Generates the 3D metabolic 'truth' volume."""
    brain_mask = (t1_data > 0)
    source_map = np.zeros_like(seg_data, dtype=np.float32)
    
    # Baseline Autofluorescence + Noise
    source_map[brain_mask] = 0.08 + np.random.normal(0, 0.01, source_map.shape)[brain_mask]
    
    # Assign Metabolic Intensities
    source_map[seg_data == 4] = 1.0  # Enhancing
    source_map[seg_data == 1] = 0.5  # Infiltrative
    source_map[seg_data == 2] = 0.1  # Necrotic
    
    # Isotropic 3D Scattering
    fluo_map_3d = gaussian_filter(source_map, sigma=1.5)
    return np.clip(fluo_map_3d, 0, 1.2)

def save_3d_fluorescence_volumes():
    """Stage 1: Processes BraTS folders and saves 3D fluo maps and masks."""
    if not os.path.exists(TO_PATH): os.makedirs(TO_PATH)

    for folder in os.listdir(FROM_PATH):
        patient_path = os.path.join(FROM_PATH, folder)
        if not os.path.isdir(patient_path): continue
            
        seg_file = os.path.join(patient_path, f"{folder}-seg.nii.gz")
        t1_file = os.path.join(patient_path, f"{folder}-t1n.nii.gz")
        if not os.path.exists(t1_file): t1_file = os.path.join(patient_path, f"{folder}-t1c.nii.gz")
        
        if os.path.exists(t1_file):
            print(f"Generating 3D Map for: {folder}")
            t1_img = nib.load(t1_file)
            t1_data = t1_img.get_fdata()
            
            if os.path.exists(seg_file):
                seg_data = nib.load(seg_file).get_fdata()
            else:
                seg_data = np.zeros_like(t1_data)

            fluo_3d = generate_3d_fluorescence_map(seg_data, t1_data)
            
            dest_folder = os.path.join(TO_PATH, folder)
            os.makedirs(dest_folder, exist_ok=True)
            
            nib.save(nib.Nifti1Image(fluo_3d, t1_img.affine, t1_img.header), 
                     os.path.join(dest_folder, f"fluo_map_3d.nii.gz"))
            nib.save(nib.Nifti1Image(seg_data.astype(np.uint8), t1_img.affine, t1_img.header), 
                     os.path.join(dest_folder, f"ground_truth_3d.nii.gz"))

def generate_camera_views(pitch, yaw, absorption_coeff=0.015):
    """Stage 2: Takes 3D maps from TO_PATH and saves 2D views to TRAINING_PATH."""
    if not os.path.exists(TRAINING_PATH): os.makedirs(TRAINING_PATH)

    for folder in os.listdir(TO_PATH):
        patient_path = os.path.join(TO_PATH, folder)
        fluo_file = os.path.join(patient_path, "fluo_map_3d.nii.gz")
        
        if os.path.exists(fluo_file):
            print(f"Capturing view for {folder} (P:{pitch}, Y:{yaw})")
            fluo_3d = nib.load(fluo_file).get_fdata()
            
            # 1. Rotate volume for camera perspective
            # order=1 is bilinear (faster for training data gen)
            rotated = rotate(fluo_3d, pitch, axes=(1, 2), reshape=False, order=1)
            rotated = rotate(rotated, yaw, axes=(0, 2), reshape=False, order=1)
            
            # 2. Apply Optical Depth Decay
            z_dim = rotated.shape[2]
            absorption = np.exp(-absorption_coeff * np.arange(z_dim))
            attenuated = rotated * absorption[np.newaxis, np.newaxis, :]
            
            # 3. Project to 2D
            view_2d = np.sum(attenuated, axis=2)
            
            # Save as NumPy or PNG (NumPy is better for training depth/AI)
            view_dest = os.path.join(TRAINING_PATH, folder)
            os.makedirs(view_dest, exist_ok=True)
            np.save(os.path.join(view_dest, f"view_p{pitch}_y{yaw}.npy"), view_2d)

# Example Execution
if __name__ == "__main__":
    #Make maps
    save_3d_fluorescence_volumes()
    
    #Generate a few specific views (e.g., Top and a 45-degree angle)
    generate_camera_views(pitch=0, yaw=0)
    generate_camera_views(pitch=45, yaw=45)
