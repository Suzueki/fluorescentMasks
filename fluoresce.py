import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import random

# --- CONFIGURATION ---
FROM_PATH = r"/content/drive/MyDrive/FluoroMask/Data/Masks-Pretreat" 
TRAINING_PATH = r"/content/drive/MyDrive/FluoroMask/DepthData"
SAMPLES_PER_PATIENT = 20

def apply_realistic_physics(sub_vol_seg, absorption_coeff=0.02, scattering_base=0.5):
    """
    Simulates light traveling back through 'slices in front'.
    Deeper slices get progressively blurrier and dimmer.
    """
    h, w, d = sub_vol_seg.shape
    accumulated_fluo = np.zeros((h, w), dtype=np.float32)

    for z in range(d):
        slice_data = (sub_vol_seg[:, :, z] > 0).astype(np.float32)
        if np.sum(slice_data) == 0: continue
        
        # 1. Beer-Lambert Dimming (Intensity)
        intensity = np.exp(-absorption_coeff * z)
        
        # 2. Scattering (Blur increases with depth z)
        # Sigma grows with z: Surface (z=0) is sharp, Depth (z=10) is blurry
        sigma = scattering_base * (z + 1)**0.5 
        blurred_slice = gaussian_filter(slice_data * intensity, sigma=sigma)
        
        accumulated_fluo += blurred_slice

    # Final normalization
    if np.max(accumulated_fluo) > 0:
        accumulated_fluo /= np.max(accumulated_fluo)
    return accumulated_fluo

def get_depth_map_from_slice(seg_mask, axis, slice_idx):
    # Re-orienting for random surgical approach
    if axis == 0: vol = seg_mask[slice_idx:, :, :]
    elif axis == 1: vol = seg_mask[:, slice_idx:, :]
    else: vol = seg_mask[:, :, slice_idx:]

    h, w, d = vol.shape
    depth_map = np.zeros((h, w), dtype=np.float32)
    tumor_mask = (vol > 0)
    has_tumor = np.any(tumor_mask, axis=2)
    depths = np.argmax(tumor_mask, axis=2)
    depth_map[has_tumor] = depths[has_tumor].astype(np.float32)
    return depth_map, vol

def generate_depth_dataset():
    if not os.path.exists(TRAINING_PATH): os.makedirs(TRAINING_PATH)

    for folder in os.listdir(FROM_PATH):
        patient_path = os.path.join(FROM_PATH, folder)
        if not os.path.isdir(patient_path): continue

        seg_file = os.path.join(patient_path, f"{folder}-seg.nii.gz")
        t1_file = os.path.join(patient_path, f"{folder}-t1n.nii.gz")
        if not os.path.exists(t1_file): t1_file = os.path.join(patient_path, f"{folder}-t1c.nii.gz")
        
        if os.path.exists(t1_file) and os.path.exists(seg_file):
            t1_data = nib.load(t1_file).get_fdata()
            seg_data = nib.load(seg_file).get_fdata()

            for i in range(SAMPLES_PER_PATIENT):
                sample_id = f"{folder}_sample_{i}"
                dest_dir = os.path.join(TRAINING_PATH, sample_id)
                if os.path.exists(dest_dir): continue

                chosen_axis = random.randint(0, 2)
                start_slice = random.randint(0, int(t1_data.shape[chosen_axis] * 0.6))

                depth_gt, sub_vol_seg = get_depth_map_from_slice(seg_data, chosen_axis, start_slice)
                
                # Filter out empty samples
                if np.sum(depth_gt) < 10: continue 

                # Apply the realistic physics loop
                fluo_2d = apply_realistic_physics(sub_vol_seg)

                # Get MRI reference
                if chosen_axis == 0: mri_ref = t1_data[start_slice, :, :]
                elif chosen_axis == 1: mri_ref = t1_data[:, start_slice, :]
                else: mri_ref = t1_data[:, :, start_slice]
                mri_ref /= (np.max(mri_ref) + 1e-8)

                os.makedirs(dest_dir, exist_ok=True)
                np.save(os.path.join(dest_dir, "fluo_2d.npy"), fluo_2d)
                np.save(os.path.join(dest_dir, "depth_gt.npy"), depth_gt)
                np.save(os.path.join(dest_dir, "mri_ref.npy"), mri_ref)
                print(f"Saved {sample_id}")

if __name__ == "__main__":
    generate_depth_dataset()