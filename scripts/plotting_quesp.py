#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 10 18:23:03 2025

@author: jonah
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_t1_map(t1_fits, reference_image, masks):
    """
    Reconstructs and plots the T1 map from pixel-wise fit results.
    This version is corrected to work with the new data format.
    """
    # Create an empty map to be filled with fitted T1 values
    t1_map = np.full(reference_image.shape, np.nan)
    
    # Reconstruct the T1 map using coordinates from the masks
    for roi_label, t1_values in t1_fits.items():
        if roi_label in masks:
            # Get the coordinates for the current ROI
            y_coords, x_coords = np.where(masks[roi_label])
            
            # Fill in the T1 map with the fitted values
            for i, (y, x) in enumerate(zip(y_coords, x_coords)):
                if i < len(t1_values):
                    t1_map[y, x] = t1_values[i]
            
    # Create a combined mask for plotting
    combined_mask = np.zeros_like(reference_image, dtype=bool)
    for mask in masks.values():
        combined_mask = np.logical_or(combined_mask, mask)

    # Plotting logic remains the same
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(reference_image, cmap='gray')
    
    t1_map_masked = np.ma.masked_where(~combined_mask, t1_map)
    overlay = ax.imshow(t1_map_masked, cmap='plasma', alpha=0.9, vmin=0, vmax=np.nanmax(t1_map))
    
    cbar = fig.colorbar(overlay, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('T₁ (ms)', fontsize=24, fontweight='bold')
    cbar.ax.tick_params(labelsize=18)
    
    ax.axis('off')
    ax.set_title('T₁ Map', fontsize=28, fontweight='bold')
    fig.tight_layout()
    st.pyplot(fig)

def plot_quesp_maps(quesp_fits, masks, reference_image):
    """
    Reconstructs and plots fb, kb, and R² maps for each fitted pool,
    using robust percentile-based colormaps to handle outliers.
    """
    if not quesp_fits:
        st.warning("No QUESP fit data available to plot.")
        return

    # Get a list of all unique chemical pools that were fitted
    first_roi_label = next(iter(quesp_fits))
    all_pool_names = list(quesp_fits[first_roi_label].keys())

    # Get the coordinates for each mask once for efficiency
    coords_by_roi = {label: np.argwhere(mask) for label, mask in masks.items()}
    
    # Create a combined mask for plotting all ROIs
    combined_mask = np.zeros_like(reference_image, dtype=bool)
    for mask in masks.values():
        combined_mask = np.logical_or(combined_mask, mask)

    # Iterate through each chemical pool to create a set of maps
    for pool_name in all_pool_names:
        st.subheader(f"Results for {pool_name}")

        # Initialize empty maps for the current pool
        fb_map = np.full(reference_image.shape, np.nan)
        kb_map = np.full(reference_image.shape, np.nan)
        r2_map = np.full(reference_image.shape, np.nan)

        # Reconstruct the full maps from the per-ROI pixel data
        for roi_label, fit_data in quesp_fits.items():
            if pool_name in fit_data:
                coords = coords_by_roi[roi_label]
                pool_fit_data = fit_data[pool_name]
                
                for i, (y, x) in enumerate(coords):
                    if i < len(pool_fit_data['fb_values']):
                        fb_map[y, x] = pool_fit_data['fb_values'][i]
                        kb_map[y, x] = pool_fit_data['kb_values'][i]
                        r2_map[y, x] = pool_fit_data['r2_values'][i]
        
        # Calculate robust color limits using percentiles to ignore outliers
        valid_fb = fb_map[combined_mask]
        fb_vmin, fb_vmax = (np.nanpercentile(valid_fb, [1, 99]) if np.any(valid_fb) else (0, 0.1))

        valid_kb = kb_map[combined_mask]
        kb_vmin, kb_vmax = (np.nanpercentile(valid_kb, [1, 99]) if np.any(valid_kb) else (0, 2000))
        
        # Plotting the three maps
        fig, axs = plt.subplots(1, 3, figsize=(24, 8))
        fig.suptitle(f'QUESP Maps for {pool_name}', fontsize=32, fontweight='bold')

        # Plot fb map
        fb_masked = np.ma.masked_where(~combined_mask, fb_map)
        axs[0].imshow(reference_image, cmap='gray')
        im0_overlay = axs[0].imshow(fb_masked, cmap='viridis', alpha=0.9, vmin=fb_vmin, vmax=fb_vmax)
        axs[0].set_title('Proton Fraction (fb)', fontsize=24, fontweight='bold')
        axs[0].axis('off')
        fig.colorbar(im0_overlay, ax=axs[0], fraction=0.046, pad=0.04).set_label('f$_{b}$')

        # Plot kb map
        kb_masked = np.ma.masked_where(~combined_mask, kb_map)
        axs[1].imshow(reference_image, cmap='gray')
        im1_overlay = axs[1].imshow(kb_masked, cmap='magma', alpha=0.9, vmin=kb_vmin, vmax=kb_vmax)
        axs[1].set_title('Exchange Rate (kb)', fontsize=24, fontweight='bold')
        axs[1].axis('off')
        fig.colorbar(im1_overlay, ax=axs[1], fraction=0.046, pad=0.04).set_label('k$_{b}$ (s⁻¹)')
        
        # Plot R² map (R² is naturally bounded between 0 and 1)
        r2_masked = np.ma.masked_where(~combined_mask, r2_map)
        axs[2].imshow(reference_image, cmap='gray')
        im2_overlay = axs[2].imshow(r2_masked, cmap='cividis', vmin=0, vmax=1)
        axs[2].set_title('Fit Quality (R²)', fontsize=24, fontweight='bold')
        axs[2].axis('off')
        fig.colorbar(im2_overlay, ax=axs[2], fraction=0.046, pad=0.04).set_label('R²')

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        st.pyplot(fig)


def calculate_quesp_stats(quesp_fits):
    """
    Calculates statistics (mean, std) for each ROI and chemical pool,
    excluding the top and bottom 1% of data to remove outliers.
    """
    stats_list = []
    
    # The structure of quesp_fits is {roi: {pool: {param: [values]}}}
    for roi_label, pools in quesp_fits.items():
        for pool_name, params in pools.items():
            
            # --- HIGHLIGHTED CHANGE ---
            # Filter each parameter list to exclude outliers before calculating stats
            
            # Filter fb_values
            fb_values = np.array(params['fb_values'])
            fb_valid = fb_values[~np.isnan(fb_values)]
            if fb_valid.size > 0:
                fb_p1, fb_p99 = np.percentile(fb_valid, [1, 99])
                fb_filtered = fb_valid[(fb_valid >= fb_p1) & (fb_valid <= fb_p99)]
                fb_mean = np.mean(fb_filtered)
                fb_std = np.std(fb_filtered)
            else:
                fb_mean, fb_std = np.nan, np.nan

            # Filter kb_values
            kb_values = np.array(params['kb_values'])
            kb_valid = kb_values[~np.isnan(kb_values)]
            if kb_valid.size > 0:
                kb_p1, kb_p99 = np.percentile(kb_valid, [1, 99])
                kb_filtered = kb_valid[(kb_valid >= kb_p1) & (kb_valid <= kb_p99)]
                kb_mean = np.mean(kb_filtered)
                kb_std = np.std(kb_filtered)
            else:
                kb_mean, kb_std = np.nan, np.nan

            # R² is already bounded, so no need to filter
            r2_mean = np.nanmean(params['r2_values'])
            # --- END CHANGE ---

            stats_list.append({
                'ROI': roi_label,
                'Pool': pool_name,
                'fb Mean': fb_mean,
                'fb Std Dev': fb_std,
                'kb Mean (s⁻¹)': kb_mean,
                'kb Std Dev (s⁻¹)': kb_std,
                'Mean R²': r2_mean
            })
            
    if not stats_list:
        return pd.DataFrame()

    # Create and format the DataFrame
    stats_df = pd.DataFrame(stats_list)
    return stats_df.set_index(['ROI', 'Pool'])