#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 16:21:48 2024

@author: jonah
"""
import os
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable

def plot_wassr_aha(session_state):
    save_path = session_state.submitted_data["save_path"]
    plot_path = os.path.join(save_path, 'Plots')
    if not os.path.isdir(plot_path):
        os.makedirs(plot_path)

    all_fits = session_state.processed_data["wassr_fits"]
    fits = {k: v for k, v in all_fits.items() if "lv" not in k.lower()}

    data = []
    for segment, b0_values in fits.items():
        for val in b0_values:
            data.append({'Segment': segment, 'B0 Shift (ppm)': val})
    df = pd.DataFrame(data)

    sns.set(style="whitegrid")
    fig, ax = plt.subplots(figsize=(9, 6))  
    palette = sns.color_palette("husl", len(df['Segment'].unique()))
    sns.boxplot(x='Segment', y='B0 Shift (ppm)', data=df, palette=palette, width=0.4, ax=ax)

    ax.set_title('B$_0$ Shift by AHA Segment', fontsize=28, fontname='Arial', weight='bold')
    ax.set_xlabel('', fontsize=18)

    ax.set_ylabel('B$_0$ Shift (ppm)', fontsize=16, fontname='Arial')
    ax.tick_params(labelsize=14)
    fig.tight_layout()

    plot_file = os.path.join(plot_path, 'WASSR_Boxplot.png')
    fig.savefig(plot_file, dpi=300)
    st.pyplot(fig) 

def plot_wassr(image, session_state):
    save_path = session_state.submitted_data["save_path"]
    image_path = os.path.join(save_path, 'Images')
    b0_image = np.zeros_like(image, dtype='float')
    if not os.path.isdir(image_path):
        os.makedirs(image_path)
    # Determine cropping bounds using LV mask
    if session_state.submitted_data['organ'] == 'Cardiac':
        lv_mask = session_state.user_geometry["masks"]["lv"]
        y_indices, x_indices = np.where(lv_mask)
        x_min, x_max = max(np.min(x_indices) - 20, 0), min(np.max(x_indices) + 20, lv_mask.shape[1])
        y_min, y_max = max(np.min(y_indices) - 20, 0), min(np.max(y_indices) + 20, lv_mask.shape[0])
        # Use AHA segments for actual plotting
        segment_masks = session_state.user_geometry["aha"]
        fits = session_state.processed_data['wassr_fits']
        # Fill in b0_image using the segment masks
        for label, coord_list in segment_masks.items():
            data = fits[label]
            for idx, (i, j) in enumerate(coord_list):
                b0_image[i, j] = data[idx]
    else:
        masks = session_state.user_geometry["masks"]
        x_min, x_max = 0, image.shape[1]
        y_min, y_max = 0, image.shape[0]
        fits = session_state.processed_data['wassr_fits']
        for label, mask in masks.items():
            data = fits[label]
            mask_indices = np.argwhere(mask)
            for idx, (i, j) in enumerate(mask_indices):
                b0_image[i, j] = data[idx]

    transparent_b0 = np.ma.masked_where(b0_image == 0, b0_image)
    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.imshow(image[y_min:y_max, x_min:x_max], cmap='gray')
    im = ax.imshow(transparent_b0[y_min:y_max, x_min:x_max], cmap='plasma', alpha=0.9)
    ax.set_title('WASSR Map', fontsize=28, fontname='Arial', weight='bold')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label('B$_0$ Shift (ppm)', fontname='Arial', fontsize=16)
    ax.axis('off')
    st.subheader("B0 Map")
    st.pyplot(fig)
    plt.savefig(os.path.join(image_path, 'B0_Map.png'), dpi=300, bbox_inches="tight")