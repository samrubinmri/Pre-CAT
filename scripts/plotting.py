#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 12:29:12 2025

@author: jonah
"""
import os
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from matplotlib.patches import Patch
from scipy.signal import medfilt2d
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import Normalize

def pixelwise_mapping(image, session_state):
    if session_state.submitted_data['organ'] == 'Cardiac':
        masks = {}
        masks["lv"] = session_state.user_geometry["masks"]["lv"]
        y_indices, x_indices = np.where(masks["lv"])
        x_min, x_max = max(np.min(x_indices) - 20, 0), min(np.max(x_indices) + 20, masks["lv"].shape[1])
        y_min, y_max = max(np.min(y_indices) - 20, 0), min(np.max(y_indices) + 20, masks["lv"].shape[0])
    else:
        masks = session_state.user_geometry["masks"]
        x_min, x_max = 0, image.shape[1] 
        y_min, y_max = 0, image.shape[0]
    fits = session_state.processed_data["pixelwise"]["fits"]
    save_path = session_state.submitted_data["save_path"]
    image_path = os.path.join(save_path, 'Images')
    if not os.path.isdir(image_path):
        os.makedirs(image_path)
    
    # Initialize empty contrast images for all ROIs combined
    mt_image = np.full_like(image, np.nan, dtype=float)
    amide_image = np.full_like(image, np.nan, dtype=float)
    creatine_image = np.full_like(image, np.nan, dtype=float)
    noe_image = np.full_like(image, np.nan, dtype=float)

    # Iterate through each ROI and populate the contrast images
    for label, mask in masks.items():
        data = fits[label]
        mt_list = [datum["Contrasts"]["MT"] for datum in data]
        amide_list = [datum["Contrasts"]["Amide"] for datum in data]
        creatine_list = [datum["Contrasts"]["Creatine"] for datum in data]
        noe_list = [datum["Contrasts"]["NOE"] for datum in data]

        mask_indices = np.argwhere(mask)
        for idx, (i, j) in enumerate(mask_indices):
            mt_image[i, j] = mt_list[idx]
            amide_image[i, j] = amide_list[idx]
            creatine_image[i, j] = creatine_list[idx]
            noe_image[i, j] = noe_list[idx]

    # Apply median filtering to smooth the contrast images
    mt_image = medfilt2d(mt_image, kernel_size=3)
    amide_image = medfilt2d(amide_image, kernel_size=3)
    creatine_image = medfilt2d(creatine_image, kernel_size=3)
    noe_image = medfilt2d(noe_image, kernel_size=3)

    # Plotting helper function
    def plot_contrast(base_image, contrast_image, title):
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(base_image[y_min:y_max,x_min:x_max], cmap="gray")
        im = ax.imshow(contrast_image[y_min:y_max,x_min:x_max], cmap="viridis", alpha=0.7, norm=Normalize(vmin=0, vmax=np.nanmax(contrast_image)))
        ax.set_title(title, fontsize=28, weight='bold', fontname='Arial')
        ax.axis("off")
        
        # Add colorbar
        cbar = fig.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label("CEST Contrast (%)", fontsize=16)
        cbar.ax.tick_params(labelsize=14)

        return fig

    # Displaying images in a 2x2 grid using Streamlit
    st.subheader("Pixelwise Maps")
    contrasts = [mt_image, amide_image, creatine_image, noe_image]
    titles = ["MT", "Amide", "Creatine", "NOE"]

    # Use containers to ensure alignment
    for i in range(0, len(contrasts), 2):  # Iterate in steps of 2
        with st.container():
            cols = st.columns(2)
            with cols[0]:
                fig = plot_contrast(image, contrasts[i], titles[i])
                plt.savefig(image_path + '/' + titles[i] + '_Contrast_Map.png', dpi=300, bbox_inches="tight")
                st.pyplot(fig)
            with cols[1]:
                fig = plot_contrast(image, contrasts[i + 1], titles[i + 1])
                plt.savefig(image_path + '/' + titles[i + 1] + '_Contrast_Map.png', dpi=300, bbox_inches="tight")
                st.pyplot(fig)
    
def show_segmentation(image, session_state):
    # Get vars from session state
    mask = session_state.user_geometry["masks"]["lv"]
    labeled_segments = session_state.user_geometry["aha"]
    save_path = session_state.submitted_data["save_path"]
    image_path = os.path.join(save_path, 'Images')
    if not os.path.isdir(image_path):
        os.makedirs(image_path)
    # Initialize an empty RGB array for the segmentation
    segmented = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    # Zoom into the region based on the mask with a margin of ±20 pixels
    y_indices, x_indices = np.where(mask)
    x_min, x_max = max(np.min(x_indices) - 20, 0), min(np.max(x_indices) + 20, mask.shape[1])
    y_min, y_max = max(np.min(y_indices) - 20, 0), min(np.max(y_indices) + 20, mask.shape[0])
    # Define segment colors
    coords = {
        'Inferoseptal': (255, 0, 0),      # red       inferoseptal
        'Anteroseptal': (0, 255, 0),      # green     anteroseptal
        'Anterior': (0, 0, 255),          # blue      anterior
        'Anterolateral': (255, 165, 0),   # orange    anterolateral
        'Inferolateral': (255, 255, 100), # yellow    inferolateral
        'Inferior': (128, 0, 128)         # purple    inferior
    }
    # Apply colors to each segment
    for segment, color in coords.items():
        for coord in labeled_segments[segment]:
            segmented[coord[0], coord[1]] = np.array(color, dtype=np.uint8)
    # Set up subplots for the original image and segmentation overlay
    fig, ax = plt.subplots(1, 1, figsize=(9, 12))
    # Display the cropped original image with segmentation overlay
    ax.imshow(image[y_min:y_max, x_min:x_max], cmap='gray')
    ax.imshow(segmented[y_min:y_max, x_min:x_max], alpha=0.5)
    # Create legend for segmentation
    legend_elements = [Patch(facecolor=np.array(color) / 255, edgecolor='black', label=label) for label, color in coords.items()]
    ax.legend(handles=legend_elements, loc="center left", bbox_to_anchor=(1, 0.5), fontsize = 24)
    # Remove axis labels
    ax.axis('off')
    st.subheader('AHA Segmentation')
    st.pyplot(fig)
    plt.savefig(image_path + '/AHA_Segmentation.png', dpi = 300, bbox_inches="tight")
    
def show_rois(image, session_state):
    masks = session_state.user_geometry["masks"]  # Retrieve the masks dictionary
    save_path = session_state.submitted_data["save_path"]
    image_path = os.path.join(save_path, 'Images')
    if not os.path.isdir(image_path):
        os.makedirs(image_path)
    colors = plt.cm.get_cmap('tab20', len(masks))  # Generate unique colors for each ROI
    segmented = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)  # Initialize RGB array for segmentation
    # Iterate through ROIs and apply colors
    for i, (label, mask) in enumerate(masks.items()):
        y_indices, x_indices = np.where(mask)
        segmented[y_indices, x_indices] = (np.array(colors(i)[:3]) * 255).astype(np.uint8)  # Apply color
    # Zoom into the region based on the combined mask with a margin of ±20 pixels
    all_y_indices, all_x_indices = np.where(np.any(list(masks.values()), axis=0))
    x_min, x_max = max(np.min(all_x_indices) - 20, 0), min(np.max(all_x_indices) + 20, image.shape[1])
    y_min, y_max = max(np.min(all_y_indices) - 20, 0), min(np.max(all_y_indices) + 20, image.shape[0])
    # Set up subplots for the original image and segmentation overlay
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    #ax.imshow(image[y_min:y_max, x_min:x_max], cmap='gray')  # Display cropped original image
    #ax.imshow(segmented[y_min:y_max, x_min:x_max], alpha=0.5)  # Overlay segmentation with transparency
    ax.imshow(image, cmap='gray') # Uncropped
    ax.imshow(segmented, alpha=0.5)
    # Create legend for ROIs
    legend_elements = [
        Patch(facecolor=np.array(colors(i)[:3]), edgecolor='black', label=label) for i, label in enumerate(masks.keys())
    ]
    ax.legend(handles=legend_elements, loc="center left", bbox_to_anchor=(1, 0.5), fontsize=12)
    # Remove axis labels
    ax.axis('off')
    # Display the figure in Streamlit
    st.subheader('ROIs')
    st.pyplot(fig)
    plt.savefig(image_path + '/ROIs.png', dpi = 300, bbox_inches="tight")

def plot_zspec(session_state):
    fits = session_state.processed_data['fits']
    roi_count = len(fits)
    
    save_path = session_state.submitted_data["save_path"]
    plot_path = os.path.join(save_path, 'Plots')
    if not os.path.isdir(plot_path):
        os.makedirs(plot_path)

    # Dynamically calculate number of rows and columns
    n_cols = 3
    n_rows = -(-roi_count // n_cols)  # Ceiling division for rows

    # Z-Spectra Plots
    st.subheader("Z-Spectra")
    for row in range(n_rows):
        cols = st.columns(min(n_cols, roi_count - row * n_cols))
        for i, roi in enumerate(list(fits.keys())[row * n_cols:(row + 1) * n_cols]):
            with cols[i]:
                fit = fits[roi]
                data_dict = fit['Data_Dict']
                OffsetsInterp = data_dict['Offsets_Interp']
                Offsets = data_dict['Offsets_Corrected']
                Spectrum = data_dict['Zspec']
                Water_Fit = data_dict['Water_Fit']
                Mt_Fit = data_dict['MT_Fit']
                Noe_Fit = data_dict['NOE_Fit']
                Creatine_Fit = data_dict['Creatine_Fit']
                Amide_Fit = data_dict['Amide_Fit']

                # Plot Z-Spectra
                fig, ax = plt.subplots(figsize=(12, 10))
                ax.plot(Offsets, Spectrum, '.', markersize=15, fillstyle='none', color='black', label="Raw")
                ax.plot(OffsetsInterp, 1 - Water_Fit, linewidth=4, color='#0072BD', label="Water")
                ax.plot(OffsetsInterp, 1 - Mt_Fit, linewidth=4, color='#EDB120', label="MT")
                ax.plot(OffsetsInterp, 1 - Noe_Fit, linewidth=4, color='#77AC30', label="NOE")
                ax.plot(OffsetsInterp, 1 - Amide_Fit, linewidth=4, color='#7E2F8E', label="Amide")
                ax.plot(OffsetsInterp, 1 - Creatine_Fit, linewidth=4, color='#A2142F', label="Creatine")
                ax.plot(OffsetsInterp, 1 - (Water_Fit + Mt_Fit + Noe_Fit + Creatine_Fit + Amide_Fit),
                        linewidth=4, color='#D95319', label="Fit")

                ax.legend(fontsize=16)
                ax.invert_xaxis()
                ax.tick_params(axis='both', which='major', labelsize=16)
                ax.set_ylim([0, 1])
                ax.set_xlabel("Offset frequency (ppm)", fontsize=18, fontname='Arial')
                ax.set_ylabel("$S/S_0$", fontsize=18, fontname='Arial')
                fig.suptitle(roi, fontsize=28, weight='bold', fontname='Arial')
                plt.grid(False)
                st.pyplot(fig)
                plt.savefig(plot_path + '/' + roi + '_Zspec.png', dpi = 300, bbox_inches="tight")

    # Lorentzian Difference Plots
    st.subheader("Lorentzian Difference Plots")
    for row in range(n_rows):
        cols = st.columns(min(n_cols, roi_count - row * n_cols))
        for i, roi in enumerate(list(fits.keys())[row * n_cols:(row + 1) * n_cols]):
            with cols[i]:
                fit = fits[roi]
                data_dict = fit['Data_Dict']
                OffsetsInterp = data_dict['Offsets_Interp']
                Offsets = data_dict['Offsets_Corrected']
                Noe_Fit = data_dict['NOE_Fit']
                Amide_Fit = data_dict['Amide_Fit']
                Creatine_Fit = data_dict['Creatine_Fit']
                Lorentzian_Difference = data_dict['Lorentzian_Difference']

                # Plot Lorentzian Difference
                fig, ax = plt.subplots(figsize=(12, 10))
                ax.fill_between(Offsets, Lorentzian_Difference * 100, 0, color='gray', alpha=0.5)
                ax.plot(OffsetsInterp, Noe_Fit * 100, linewidth=4, color='#77AC30', label="NOE")
                ax.plot(OffsetsInterp, Amide_Fit * 100, linewidth=4, color='#7E2F8E', label="Amide")
                ax.plot(OffsetsInterp, Creatine_Fit * 100, linewidth=4, color='#A2142F', label="Creatine")

                ax.legend(fontsize=16)
                ax.invert_xaxis()
                ax.tick_params(axis='both', which='major', labelsize=16)
                ax.set_xlabel("Offset frequency (ppm)", fontsize=18, fontname='Arial')
                ax.set_ylabel("CEST Contrast (%)", fontsize=18, fontname='Arial')
                fig.suptitle(roi, fontsize=28, weight='bold', fontname='Arial')
                plt.grid(False)
                st.pyplot(fig)
                plt.savefig(plot_path + '/' + roi + '_Lorentzian_Dif.png', dpi = 300, bbox_inches="tight")