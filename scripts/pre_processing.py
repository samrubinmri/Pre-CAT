#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  9 15:07:25 2025

@author: jonah
"""
import numpy as np
import streamlit as st
from scipy.ndimage import uniform_filter1d
from sklearn.decomposition import PCA
import scripts.BrukerMRI as bruker 
from bart import bart 

# Tunable post-processing parameters
SPIKE_THRESHOLD_STD = 0.5
MOVING_AVG_WINDOW = 5

# ==============================================================================
# Helper Function for Reconstruction (used by other functions)
# ==============================================================================
def recon(ksp, traj):
    """Function for reconstructing a single image using BART."""
    img = bart(1, 'nufft -i', traj, ksp)
    img = bart(1, 'rss 8', img)
    img = np.abs(img)
    img = np.squeeze(img)
    return img

# ==============================================================================
# Motion Correction Function
# ==============================================================================
def motion_correction(ksp, traj, method, experiment_type):
    """
    Performs motion correction by identifying and deleting corrupted segments.
    
    Args:
        ksp (np.array): The raw k-space data (points, spokes, coils, offsets).
        traj (np.array): The trajectory data (3, points, spokes).
        method (dict): The experiment's method dictionary.
        
    Returns:
        np.array: The motion-corrected image stack.
    """
    print("Starting motion correction...")
    
    # --- Get Parameters ---
    points, n_spokes, n_coils, n_offsets = ksp.shape
    seg = method['Num_Traj_per_Seg']
    # Corrected ppm calculation to match previous working versions
    offsets_ppm = np.round(method["Cest_Offsets"] / (method["PVM_FrqWork"][0]), 2)
    n_segments = n_spokes // seg
    
    # Region based on experiment type
    if experiment_type == 'cest':
        ranges = [(-4.0, -1.4), (1.4, 4.0)]
        # Assign to 'indices' for consistency
        indices = np.where(
            (offsets_ppm >= ranges[0][0]) & (offsets_ppm <= ranges[0][1]) |
            (offsets_ppm >= ranges[1][0]) & (offsets_ppm <= ranges[1][1])
        )[0]
    elif experiment_type == 'wassr':
        # Assign to 'indices' for consistency
        indices = np.arange(n_offsets)
    
    counts = {}
    for index in indices:
        coil_spike_info = []
        for coil in range(n_coils):
            proj_coil = np.fft.fftshift(np.fft.fft(ksp[:, :, coil, index], axis=0), axes=0)
            mag_proj_coil = np.abs(proj_coil)
            reshaped_proj = mag_proj_coil[:, :n_segments * seg].reshape((points, n_segments, seg))
            segment_totals_coil = np.sum(reshaped_proj, axis=(0, 2))
            moving_avg = uniform_filter1d(segment_totals_coil, size=MOVING_AVG_WINDOW, mode='nearest')
            std_dev = np.std(segment_totals_coil - moving_avg)
            is_spike = segment_totals_coil < (moving_avg - SPIKE_THRESHOLD_STD * std_dev) if std_dev > 0 else np.zeros_like(segment_totals_coil, dtype=bool)
            total_magnitude = np.sum(moving_avg[is_spike] - segment_totals_coil[is_spike]) if np.any(is_spike) else 0
            coil_spike_info.append({'total_magnitude': total_magnitude, 'num_spikes': np.sum(is_spike)})
        
        best_coil_info = max(coil_spike_info, key=lambda x: x['total_magnitude'])
        counts[index] = best_coil_info['num_spikes']

    N_to_remove = max(counts.values()) if counts else 0
    st.warning(f"Motion correction will remove {N_to_remove} segments from each offset.")

    # --- Apply Filtering to All Offsets ---
    filtered_images_list = []
    loading_bar = st.progress(0, text="Applying motion correction and reconstructing...")
    
    for offset_idx in range(n_offsets):
        # Find the best coil for this offset to determine which segments to remove
        coil_spike_info_mc = []
        for coil in range(n_coils):
            proj_coil = np.fft.fftshift(np.fft.fft(ksp[:, :, coil, offset_idx], axis=0), axes=0)
            mag_proj_coil = np.abs(proj_coil)
            reshaped_proj = mag_proj_coil[:, :n_segments * seg].reshape((points, n_segments, seg))
            segment_totals_coil = np.sum(reshaped_proj, axis=(0, 2))
            moving_avg = uniform_filter1d(segment_totals_coil, size=MOVING_AVG_WINDOW, mode='nearest')
            severity = moving_avg - segment_totals_coil
            severity[severity < 0] = 0
            coil_spike_info_mc.append({'coil_index': coil, 'severity_data': severity})
        
        best_coil_idx = max(coil_spike_info_mc, key=lambda x: np.sum(x['severity_data']))['coil_index']
        final_severity = next(item['severity_data'] for item in coil_spike_info_mc if item['coil_index'] == best_coil_idx)
        
        indices_of_worst_segments = np.argsort(final_severity)[-N_to_remove:] if N_to_remove > 0 else []
        
        # Delete the identified segments
        spokes_to_delete = []
        for seg_idx in indices_of_worst_segments:
            spokes_to_delete.extend(np.arange(seg_idx * seg, (seg_idx + 1) * seg))
        spokes_to_delete.sort()

        ksp_single_offset = ksp[..., offset_idx]
        ksp_deleted = np.delete(ksp_single_offset, spokes_to_delete, axis=1)
        traj_deleted = np.delete(traj, spokes_to_delete, axis=2)
        
        ksp_for_recon = np.expand_dims(ksp_deleted, axis=0)
        filtered_img_single = recon(ksp_for_recon, traj_deleted)
        filtered_images_list.append(filtered_img_single)
        
        progress = (offset_idx + 1) / n_offsets
        loading_bar.progress(progress, text=f"Applying motion correction: {offset_idx + 1}/{n_offsets}")

    loading_bar.progress(1.0, text="Motion correction complete.")
    return np.stack(filtered_images_list, axis=-1)

# ==============================================================================
# Denoising Function
# ==============================================================================
def denoise_data(image_stack):
    """
    Denoises a stack of images using Global PCA.
    
    Args:
        image_stack (np.array): A 3D array of images (height, width, n_offsets).
        
    Returns:
        np.array: The denoised 3D image stack.
    """
    
    # --- Reshape the data for PCA ---
    height, width, n_offsets_s = image_stack.shape
    data_matrix = image_stack.reshape((height * width, n_offsets_s))

    # --- Apply PCA and find optimal components ---
    pca = PCA()
    pca.fit(data_matrix)
    eigenvalues = pca.explained_variance_
    n_samples_m, n_features_n = data_matrix.shape
    
    indicator_values = []
    for i in range(1, n_features_n):
        numerator = np.sum(eigenvalues[i:])
        denominator = n_samples_m * ((n_features_n - (i-1))**5)
        indicator_values.append(np.sqrt(numerator / denominator) if denominator > 1e-9 else np.inf)
        
    n_components_to_keep = np.argmin(indicator_values) + 1
    st.warning(f"  Denoising with {n_components_to_keep} components.")

    # --- Filter data using the optimal number of components ---
    pca_denoising = PCA(n_components=n_components_to_keep)
    transformed_data = pca_denoising.fit_transform(data_matrix)
    denoised_data_matrix = pca_denoising.inverse_transform(transformed_data)
    
    # --- Reshape back to an image stack and return ---
    return denoised_data_matrix.reshape((height, width, n_offsets_s))

# ==============================================================================
# Main Post-Processing Function (Example Usage)
# ==============================================================================
def pre_processing(session_state, experiment_type):
    # This is an example of how you might structure your main function
    # Get variables from session state
    directory = session_state.submitted_data.get("folder_path")
    if experiment_type == "cest":
        num_exp = session_state.submitted_data.get("cest_path")
    elif experiment_type == "wassr":
        num_exp = session_state.submitted_data.get("wassr_path")
    
    # Load data
    exp = bruker.ReadExperiment(directory, num_exp)
    ksp = exp.GenerateKspace()
    traj = exp.traj
    method = exp.method
    offsets = np.round(method["Cest_Offsets"] / (method["PVM_FrqWork"][0]), 2)
    

    # --- Run the Pipeline ---
    # accounts for PCA and Motion Correction to be independant of each other
    if experiment_type == 'cest' and session_state.submitted_data["moco_cest"] == True  and session_state.submitted_data["pca"] == True :
        motion_corrected_stack = motion_correction(ksp, traj, method, experiment_type)
        final_denoised_stack = denoise_data(motion_corrected_stack)
        study = {"imgs": final_denoised_stack, "offsets": offsets}
    elif experiment_type == 'cest' and session_state.submitted_data["moco_cest"] == True and session_state.submitted_data["pca"] == False:
        motion_corrected_stack = motion_correction(ksp, traj, method, experiment_type)
        study = {"imgs": motion_corrected_stack, "offsets": offsets}
    elif experiment_type == 'cest' and session_state.submitted_data["moco_cest"] == False and session_state.submitted_data["pca"] == True:
        # No motion correction, but denoising: need to reconstruct per-offset
        points, n_spokes, n_coils, n_offsets = ksp.shape
        recon_stack = []
        for offset_idx in range(n_offsets):
            ksp_offset = ksp[:, :, :, offset_idx]
            traj_offset = traj  # assuming same traj for all offsets
            ksp_for_recon = np.expand_dims(ksp_offset, axis=0)
            img = recon(ksp_for_recon, traj_offset)
            recon_stack.append(img)
        recon_stack = np.stack(recon_stack, axis=-1)
        denoised_image_stack = denoise_data(recon_stack)
        study = {"imgs": denoised_image_stack, "offsets": offsets}
    return study