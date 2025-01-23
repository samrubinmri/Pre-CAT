#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 11:44:38 2024

@author: jonah
"""
import os
import sys
import time

if 'BART_TOOLBOX_PATH' in os.environ and os.path.exists(os.environ['BART_TOOLBOX_PATH']):
	sys.path.append(os.path.join(os.environ['BART_TOOLBOX_PATH'], 'python'))
elif 'TOOLBOX_PATH' in os.environ and os.path.exists(os.environ['TOOLBOX_PATH']):
	sys.path.append(os.path.join(os.environ['TOOLBOX_PATH'], 'python'))
else:
	raise RuntimeError("BART_TOOLBOX_PATH is not set correctly!")

from bart import bart
import numpy as np
import streamlit as st

import scripts.BrukerMRI as bruker
import matplotlib.pyplot as plt
from scipy.interpolate import interpn
    
def recon_bruker(num, directory):
    data = bruker.ReadExperiment(directory, num)
    offsets = np.round(data.method["Cest_Offsets"]/data.method["PVM_FrqWork"][0],1)
    imgs = data.proc_data
    imgs = np.rot90(imgs, k=2)
    study = {"imgs": imgs, "offsets": offsets}
    return study

def recon_bart(num, directory):
    data = bruker.ReadExperiment(directory, num)
    imgs = []
    offsets = np.round(data.method["Cest_Offsets"] / data.method["PVM_FrqWork"][0], 2)
    traj = data.traj
    ksp = data.GenerateKspace()
    loading_bar = st.progress(0, text="Reconstructing images...")
    for i in range(len(offsets)):
        offset_ksp = ksp[:, :, :, i]
        offset_ksp = np.expand_dims(offset_ksp, axis=0)
        img = bart(1, 'nufft -i', traj, offset_ksp)
        img = bart(1, 'rss 8', img)
        img = np.abs(img)
        imgs.append(img)
        # Update progress bar and text
        loading_bar.progress((i + 1) / len(offsets), text="Reconstructing images...")
        if i+1 == len(offsets):
            loading_bar.progress((i + 1) / len(offsets), text="Reconstruction complete.")
    imgs = np.stack(imgs, axis=2)
    study = {"imgs": imgs, "offsets": offsets}
    return study

def rotate_imgs(session_state):
    imgs = session_state.recon['cest']['imgs']  # Assuming the CEST data is in the first index
    # Stage 1: Select rotation
    if session_state['rotation_stage'] == 'select_rotation':
        fig = plt.figure()
        ax = plt.axes()
        ax.imshow(imgs[:, :, 0], cmap='gray')
        ax.axis('off')
        st.pyplot(fig, use_container_width=False)
        session_state['selected_rotation'] = st.selectbox(
            'Select the number of 90-degree counterclockwise rotations to align ventral (top) to dorsal (bottom):',
            [0, 1, 2, 3],
            index=session_state['selected_rotation']
        )
        if st.button('Rotate'):
            session_state['rotated_imgs'] = np.rot90(imgs, k=session_state['selected_rotation'], axes=(0, 1))
            session_state['rotation_stage'] = 'confirm_rotation'
            st.rerun()
            # Proceed without rerun; the next steps will be automatically updated based on session state.
    # Stage 2: Confirm rotation
    elif session_state['rotation_stage'] == 'confirm_rotation':
        fig, ax = plt.subplots()
        ax.imshow(session_state['rotated_imgs'][:, :, 0], cmap='gray')
        ax.axis('off')
        st.pyplot(fig)
        rotation_ok = st.radio('Is this rotation correct?', ['Yes', 'No'], index=0)
        if st.button('Submit Rotation'):
            if rotation_ok == 'Yes':
                # Update the session state to indicate that rotation is finalized
                session_state['rotation_stage'] = 'finalized'
                session_state.rot_done = True
                # Save the rotated images to the study
                imgs = session_state['rotated_imgs']
                session_state.recon['cest']['imgs'] = imgs  # Finalize the rotation and save the rotated images
                # No need to return the study here; this step can be handled elsewhere in the flow.
                st.write("Rotation finalized!")
                st.rerun()
            elif rotation_ok == 'No':
                # Reset the rotation stage to allow for re-selection of rotation
                session_state['rotation_stage'] = 'select_rotation'
                session_state['rotated_imgs'] = None
                st.write("Rotation not correct. Please try again.")
                st.rerun()
                # Re-run the step to allow the user to select a new rotation without rerunning the entire app.

def thermal_drift(session_state):
    THRESHOLD_PPM = 15
    images = session_state.recon['cest']['imgs']
    offsets = session_state.recon['cest']['offsets']
    # Find reference index
    ref_index = np.where(offsets > THRESHOLD_PPM)[0]
    # Apply normalization
    m0 = images[:,:,ref_index]
    offsets = np.delete(offsets, ref_index)
    images = np.delete(images, ref_index, axis=2)
    
    if np.size(ref_index) > 1:
        step = ref_index[1]-1
        ref_offsets = np.concatenate(([offsets[0]], offsets[step-1::step], [offsets[-1]]))

        matrix = np.size(images, 0)
        grid_index = np.arange(0,matrix)

        points = (grid_index,grid_index,ref_offsets)
        xi, yi, fi = np.meshgrid(grid_index, grid_index, offsets, indexing='ij')
        values = np.stack((xi, yi, fi), axis=-1)

        m0_interp = interpn(points, m0, values)
        images = np.nan_to_num(images/m0_interp)
        
        session_state.recon['cest']['imgs'] = images
        session_state.recon['cest']['offsets'] = offsets
        session_state.recon['cest']['m0'] = m0[:, :, 0]
        session_state.recon['cest']['m0_interp'] = m0_interp
        session_state.drift_done = True
        
    else:
        images = np.nan_to_num(images/m0)
        session_state.recon['cest']['imgs'] = images
        session_state.recon['cest']['offsets'] = offsets
        session_state.recon['cest']['m0'] = np.squeeze(m0[:, :, 0])
        session_state.drift_done = True