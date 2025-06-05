#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 12:29:12 2025

@author: jonah
"""
import os
import streamlit as st
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import ndimage

def plot_damb1(session_state):
    b1_fits = session_state.processed_data['b1_fits']
    if 'WASSR' and 'CEST' not in session_state.submitted_data['selection']:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))    
        im = ax.imshow(b1_fits, cmap='plasma')
        ax.set_title('$B_1$ Map', fontsize=22, fontname='Arial', weight='bold')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(im, cax=cax)
        cbar.ax.tick_params(labelsize=18)
        cbar.set_label('$\Delta\\theta$ (°)', fontsize=18)
        ax.axis('off')
        st.pyplot(fig)
    else:
        reference = 'cest' if 'CEST' in session_state.submitted_data['selection'] else 'wassr'
        ref_img = session_state.recon[reference]['m0']
        zoom_factors = (
            ref_img.shape[0] / b1_fits.shape[0],
            ref_img.shape[1] / b1_fits.shape[1]
        )
        b1_interp = ndimage.zoom(b1_fits, zoom=zoom_factors, order=1)
        if session_state.submitted_data['organ'] == 'Cardiac':
            lv_mask = session_state.user_geometry["masks"]["lv"]
            y_indices, x_indices = np.where(lv_mask)
            x_min, x_max = max(np.min(x_indices) - 20, 0), min(np.max(x_indices) + 20, lv_mask.shape[1])
            y_min, y_max = max(np.min(y_indices) - 20, 0), min(np.max(y_indices) + 20, lv_mask.shape[0])
            b1_interp *= lv_mask
        elif session_state.submitted_data['organ'] == 'Other':
            masks = session_state.user_geometry['masks']
            y_min, y_max = 0, b1_interp.shape[0]
            x_min, x_max = 0, b1_interp.shape[1]

        transparent_b1 = np.ma.masked_where(b1_interp == 0, b1_interp)
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        fig.suptitle('$B_1$ Map Visualization', fontsize=26, fontname='Arial', weight='bold')

        im0 = axs[0].imshow(b1_fits, cmap='plasma')
        axs[0].set_title('Raw $B_1$', fontsize=20, fontname='Arial', weight='bold')
        axs[0].axis('off')

        axs[1].imshow(ref_img[y_min:y_max, x_min:x_max], cmap='gray')
        im1 = axs[1].imshow(transparent_b1[y_min:y_max, x_min:x_max], cmap='plasma')
        axs[1].set_title('Interpolated on Reference', fontsize=20, fontname='Arial', weight='bold')
        axs[1].axis('off')

        divider = make_axes_locatable(axs[1])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(im1, cax=cax)
        cbar.ax.tick_params(labelsize=18)
        cbar.set_label('$\Delta\\theta$ (°)', fontsize=18)

        fig.tight_layout(rect=[0, 0, 1, 0.95])
        st.pyplot(fig)

def plot_damb1_aha(session_state):
    b1_fits = session_state.processed_data['b1_fits']
    reference = 'cest' if 'CEST' in session_state.submitted_data['selection'] else 'wassr'
    ref_img = session_state.recon[reference]['m0']
    zoom_factors = (
        ref_img.shape[0] / b1_fits.shape[0],
        ref_img.shape[1] / b1_fits.shape[1]
    )
    b1_interp = ndimage.zoom(b1_fits, zoom=zoom_factors, order=1)
    # Use AHA segments for actual plotting
    segment_masks = session_state.user_geometry["aha"]
    data = []
    for segment, coord_list in segment_masks.items():
        for (i, j) in coord_list:
            val = b1_interp[i, j]
            data.append({'Segment': segment, 'Flip Angle Error (°)': val})

    df = pd.DataFrame(data)

    # Plot with Seaborn
    sns.set(style="whitegrid")
    fig, ax = plt.subplots(figsize=(9, 6))
    palette = sns.color_palette("husl", len(df['Segment'].unique()))
    sns.boxplot(x='Segment', y='Flip Angle Error (°)', data=df, palette=palette, width=0.4, ax=ax)

    ax.set_title('Flip Angle Error by AHA Segment', fontsize=28, fontname='Arial', weight='bold')
    ax.set_xlabel('', fontsize=18)
    ax.set_ylabel('Flip Angle Error (°)', fontsize=16, fontname='Arial')
    ax.tick_params(labelsize=14)
    fig.tight_layout()

    st.pyplot(fig)


