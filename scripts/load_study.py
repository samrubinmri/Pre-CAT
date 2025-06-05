#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 11:44:38 2024

@author: jonah
"""
import os
import sys

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

def load_bruker_img(num, directory):
    data = bruker.ReadExperiment(directory, num)
    imgs = data.proc_data
    imgs = np.rot90(imgs, k=2)
    return imgs
    
def recon_bruker(num, directory):
    data = bruker.ReadExperiment(directory, num)
    offsets = np.round(data.method["Cest_Offsets"]/data.method["PVM_FrqWork"][0],1)
    imgs = data.proc_data
    visu = data.visu
    orientation = np.array(visu['VisuCoreOrientation'].reshape(3,3))
    flip_y = orientation[0, 1] < 0
    if flip_y == True:
        imgs = np.flip(imgs, axis=1)
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

def recon_damb1(session_state):
    directory = st.session_state.submitted_data['folder_path']
    theta_path = st.session_state.submitted_data['theta_path']
    two_theta_path = st.session_state.submitted_data['two_theta_path']
    exp_theta = bruker.ReadExperiment(directory, theta_path)
    exp_two_theta = bruker.ReadExperiment(directory, two_theta_path)
    theta = np.squeeze(exp_theta.proc_data)
    two_theta = np.squeeze(exp_two_theta.proc_data)
    visu = exp_theta.visu
    flip = exp_theta.acqp['ACQ_flip_angle']
    orientation = np.array(visu['VisuCoreOrientation'].reshape(3,3))
    flip_y = orientation[0, 1] < 0
    if flip_y == True:
        theta = np.flip(theta, axis=1)
        two_theta = np.flip(two_theta, axis=1)
    imgs = np.stack([theta, two_theta], axis=-1)
    study = {"imgs": imgs, "nominal_flip": flip}
    return study

def rotate_imgs(session_state, exp_type):
    imgs = session_state.recon[exp_type]['imgs']
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
            index=session_state['selected_rotation'],
            key=f"rotation_select_{exp_type}"  # Added key
        )
        if st.button('Rotate', key=f"rotate_button_{exp_type}"): # Added key
            session_state['rotated_imgs'] = np.rot90(imgs, k=session_state['selected_rotation'], axes=(0, 1))
            session_state['rotation_stage'] = 'confirm_rotation'
            st.rerun()
            
    # Stage 2: Confirm rotation
    elif session_state['rotation_stage'] == 'confirm_rotation':
        fig, ax = plt.subplots()
        ax.imshow(session_state['rotated_imgs'][:, :, 0], cmap='gray') # Corrected this line
        ax.axis('off')
        st.pyplot(fig)
        rotation_ok = st.radio(
            'Is this rotation correct?', 
            ['Yes', 'No'], 
            index=0, 
            key=f"rotation_confirm_{exp_type}" # Added key
        )
        if st.button('Submit Rotation', key=f"submit_rotation_{exp_type}"): # Added key
            if rotation_ok == 'Yes':
                session_state['rotation_stage'] = 'finalized'
                session_state.rot_done = True
                imgs = session_state['rotated_imgs']
                if exp_type == 'cest':
                    session_state.recon['cest']['imgs'] = imgs 
                elif exp_type == 'wassr':
                    session_state.recon['wassr']['imgs'] = imgs
                elif exp_type == 'damb1':
                    session_state.recon['damb1']['imgs'] = imgs
                st.write("Rotation finalized!")
                st.rerun()
            elif rotation_ok == 'No':
                session_state['rotation_stage'] = 'select_rotation'
                session_state['rotated_imgs'] = None
                st.write("Rotation not correct. Please try again.")
                st.rerun()

def quick_rot(session_state, exp_type):
    if exp_type == 'wassr':
        imgs = session_state.recon['wassr']['imgs']
        session_state.recon['wassr']['imgs'] = np.rot90(imgs, k=session_state['selected_rotation'], axes=(0,1))
    if exp_type == 'damb1':
        imgs = session_state.recon['damb1']['imgs']
        session_state.recon['damb1']['imgs'] = np.rot90(imgs, k=session_state['selected_rotation'], axes=(0,1))

def thermal_drift(session_state, exp_type):
    THRESHOLD_PPM = 15
    images = session_state.recon[exp_type]['imgs']
    offsets = session_state.recon[exp_type]['offsets']
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
        points = (grid_index, grid_index, ref_offsets)
        xi, yi, fi = np.meshgrid(grid_index, grid_index, offsets, indexing='ij')
        values = np.stack((xi, yi, fi), axis=-1)
        

        m0_interp = interpn(points, m0, values)
        images = np.nan_to_num(images/m0_interp)
        
        session_state.recon[exp_type]['imgs'] = images
        session_state.recon[exp_type]['offsets'] = offsets
        session_state.recon[exp_type]['m0'] = m0[:, :, 0]
        session_state.recon[exp_type]['m0_final'] = m0[:, :, -1]
        session_state.recon[exp_type]['m0_interp'] = m0_interp
        session_state.drift_done[exp_type] = True
        
    else:
        images = np.nan_to_num(images/m0)
        session_state.recon[exp_type]['imgs'] = images
        session_state.recon[exp_type]['offsets'] = offsets
        session_state.recon[exp_type]['m0'] = np.squeeze(m0[:, :, 0])
        session_state.drift_done[exp_type] = True