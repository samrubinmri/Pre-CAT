#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 12:35:44 2025

@author: jonah
"""

import streamlit as st
import os
from scripts import load_study, draw_rois, cest_fitting, wassr, misc
from custom import st_functions
import pickle as pkl
import pyautogui

if "is_submitted" not in st.session_state:
    st.session_state.is_submitted = False
if "submitted_data" not in st.session_state:
    st.session_state.submitted_data = {}
if "processing_active" not in st.session_state:
    st.session_state.processing_active = False
    
def clear_session_state():
    """Clear all session state."""
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    
def reset_submission_state():
    st.session_state.is_submitted = False
    st.session_state.processing_active = False
    st.session_state.submitted_data = {}
    
def validate_radial(path):
    """Check the existence of required files in a radial experiment."""
    required_files = ['method', 'acqp', 'traj', 'fid']
    missing = [file for file in required_files if not os.path.isfile(os.path.join(path, file))]
    return missing

def validate_rectilinear(path):
    """Check the existence of required files for rectilinear acquisition."""
    required_files = ['method', 'acqp']
    missing = [file for file in required_files if not os.path.isfile(os.path.join(path, file))]
    pdata_path = os.path.join(path, 'pdata')
    if not os.path.isdir(pdata_path):
        missing.append("pdata folder")
    else:
        subfolders = [f for f in os.listdir(pdata_path) if os.path.isdir(os.path.join(pdata_path, f))]
        if not subfolders:
            missing.append("subfolder within pdata")
        else:
            subfolder_path = os.path.join(pdata_path, subfolders[0])
            if not os.path.isfile(os.path.join(subfolder_path, '2dseq')):
                missing.append("2dseq file within pdata subfolder")
    return missing

hoverable_pre_cat = st_functions.add_hoverable_title_with_image_inline(
    "Pre-CAT",  # The title text
    "https://i.ibb.co/PgCH1Tg/Subject-2.png"  # Replace with your image URL
)

# Combine the static title and hoverable title into one header
st.markdown(
    f"<h1 style='font-size: 3rem; font-weight: bold;'>Welcome to {hoverable_pre_cat}</h1>",
    unsafe_allow_html=True
)

# Add description text
st.write("### A preclinical CEST-MRI analysis tool.")
with st.sidebar:
    st.write("""This webapp is associated with the following paper, please cite this work when using **Pre-CAT**.
## Citation
Weigand-Whittier J, Wendland M, Lam B, et al. *Ungated, plug-and-play cardiac CEST-MRI using radial FLASH with segmented saturation*. Magn Reson Med (2024). 10.1002/mrm.30382""")


with st.expander("Load data", expanded = not st.session_state.is_submitted):
    options = ["CEST", "WASSR", "DAMB1"]
    organs = ["Cardiac", "Other"]
    col1, col2 = st.columns((1,1))
    with col1:
        selection = st.pills("Experiment types", options, selection_mode="multi")
    with col2:
        anatomy = st.pills("Organ of interest", organs)
    
    if selection and anatomy:
        folder_path = st.text_input('Input data path', placeholder='User/Documents/MRI_Data/Project/Scan_ID')
        save_path = st.text_input('Save processed data as', placeholder='Liver_5uT_Radial')
        
        # Initialize validation flags
        cest_validation = True
        wassr_validation = True
        damb1_validation = True
        all_fields_filled = True  # Flag to track if all fields are filled
    
        if folder_path and os.path.isdir(folder_path):
            # Ensure Data folder exists within the main path
            data_folder = os.path.join(folder_path, "Data")
            if not os.path.isdir(data_folder):
                os.makedirs(data_folder)
            
            # Create folder for save_path within Data
            save_full_path = os.path.join(data_folder, save_path)
            if not os.path.isdir(save_full_path):
                os.makedirs(save_full_path)
            save_path = save_full_path  # Overwrite save_path with the full path
            
            # CEST validation
            if "CEST" in selection:
                cest_path = st.text_input('Input CEST experiment number', placeholder='5')
                if not cest_path:
                    all_fields_filled = False  # CEST path is required
                if cest_path:
                    cest_type = st.radio('CEST acquisition type', ["Radial", "Rectilinear"], horizontal=True)
                    if not cest_type:
                        all_fields_filled = False  # CEST acquisition type is required
                    cest_full_path = os.path.join(folder_path, cest_path)
                    if os.path.isdir(cest_full_path):
                        if cest_type == "Rectilinear" and "traj" in os.listdir(cest_full_path):
                            st.warning("The presence of a gradient trajectory file suggests the data might be radial. Please verify your acquisition type.")
                            cest_validation = False
                        missing_items = validate_radial(cest_full_path) if cest_type == "Radial" else validate_rectilinear(cest_full_path)
                        if missing_items:
                            st.error(f"CEST folder is missing the following required items: {', '.join(missing_items)}")
                            cest_validation = False
                        if cest_validation:
                            st.success("CEST folder validation successful!")
                    else:
                        st.error(f"CEST folder does not exist: {cest_full_path}")
                        cest_validation = False
    
            # WASSR validation
            if "WASSR" in selection:
                wassr_path = st.text_input('Input WASSR experiment number')
                if not wassr_path:
                    all_fields_filled = False  # WASSR path is required
                if wassr_path:
                    wassr_type = st.radio('WASSR acquisition type', ["Radial", "Rectilinear"], horizontal=True)
                    if not wassr_type:
                        all_fields_filled = False  # WASSR acquisition type is required
                    wassr_full_path = os.path.join(folder_path, wassr_path)
                    if os.path.isdir(wassr_full_path):
                        if wassr_type == "Rectilinear" and "traj" in os.listdir(wassr_full_path):
                            st.warning("The presence of a gradient trajectory file suggests the data might be radial.")
                            wassr_validation = False
                        missing_items = validate_radial(wassr_full_path) if wassr_type == "Radial" else validate_rectilinear(wassr_full_path)
                        if missing_items:
                            st.error(f"WASSR folder is missing the following required items: {', '.join(missing_items)}")
                            wassr_validation = False
                        if wassr_validation:
                            st.success("WASSR folder validation successful!")
                    else:
                        st.error(f"WASSR folder does not exist: {wassr_full_path}")
                        wassr_validation = False
    
            # DAMB1 validation
            if "DAMB1" in selection:
                theta_path = st.text_input('Input DAMB1 experiment number for α')
                two_theta_path = st.text_input('Input DAMB1 experiment number for 2α')
                if not theta_path or not two_theta_path:
                    all_fields_filled = False  # Both DAMB1 paths are required
    
                if theta_path and two_theta_path:
                    theta_full_path = os.path.join(folder_path, theta_path)
                    two_theta_full_path = os.path.join(folder_path, two_theta_path)
    
                    if os.path.isdir(theta_full_path):
                        theta_missing_items = validate_rectilinear(theta_full_path)
                        if theta_missing_items:
                            st.error(f"DAMB1 α folder is missing the following required items: {', '.join(theta_missing_items)}")
                            damb1_validation = False
                        else:
                            st.success("DAMB1 α folder validation successful!")
                    else:
                        st.error(f"DAMB1 α folder does not exist: {theta_full_path}")
                        damb1_validation = False
    
                    if os.path.isdir(two_theta_full_path):
                        two_theta_missing_items = validate_rectilinear(two_theta_full_path)
                        if two_theta_missing_items:
                            st.error(f"DAMB1 2α folder is missing the following required items: {', '.join(two_theta_missing_items)}")
                            damb1_validation = False
                        else:
                            st.success("DAMB1 2α folder validation successful!")
                    else:
                        st.error(f"DAMB1 2α folder does not exist: {two_theta_full_path}")
                        damb1_validation = False
            
            # Check if all fields are filled before enabling submit
            if all_fields_filled and (cest_validation and wassr_validation and damb1_validation):
                if st.button("Submit"):
                    st.success(f"Moving to next steps! Data will be saved in: {save_path}")
                    st.session_state.is_submitted = True
                    st.session_state.submitted_data = {
                        "folder_path": folder_path,
                        "save_path": save_path,
                        "selection": selection,
                        "organ": anatomy}
                    if "CEST" in selection:
                        st.session_state.submitted_data['cest_path'] = cest_path
                        st.session_state.submitted_data['cest_type'] = cest_type
                    if "WASSR" in selection: 
                        st.session_state.submitted_data['wassr_path'] = wassr_path
                        st.session_state.submitted_data['wassr_type'] = wassr_type
                    if "DAMB1" in selection:
                        st.session_state.submitted_data['theta_path'] = theta_path
                        st.session_state.submitted_data['two_theta_path'] = two_theta_path
                        
                    st.rerun()
            else:
                if not all_fields_filled:
                    st.error("Please fill in all the required fields before submitting.")
        else:
            if folder_path:
                st.error(f"The provided data path does not exist: {folder_path}")
                
if st.session_state.is_submitted:
    st.session_state.processing_active = True
    with st.expander("Process data", expanded = st.session_state.processing_active):
        # Set new session vars
        if "recon" not in st.session_state:
            st.session_state.recon = {
                "cest": None,
                "wassr": None,
                "damb1": None}
        if "user_geometry" not in st.session_state:
            st.session_state.user_geometry = {
                "rotations": None,
                "rois": None,
                "masks": None}
        if "rot_done" not in st.session_state:
            st.session_state.rot_done = False
        if "drift_done" not in st.session_state:
            st.session_state.drift_done = False
        if "rois_done" not in st.session_state:
            st.session_state.rois_done = False
        # Retrieve submitted data
        submitted_data = st.session_state.submitted_data
        # Logic for each experiment type
        if "CEST" in submitted_data["selection"]:
            cest_path = submitted_data.get("cest_path")  # Retrieve cest_path from submitted data
            cest_type = submitted_data.get("cest_type")  # Retrieve cest_type from submitted data
            folder_path = submitted_data["folder_path"]
            if cest_type == 'Rectilinear':
                if st.session_state.recon['cest'] is None:
                    data_cest = load_study.recon_bruker(cest_path, folder_path)
                    st.session_state.recon['cest'] = data_cest
            elif cest_type == 'Radial':
                if 'rotation_stage' not in st.session_state:
                    st.session_state['rotation_stage'] = 'select_rotation'  # Stages: 'select_rotation', 'confirm_rotation', 'finalized'
                if 'selected_rotation' not in st.session_state:
                    st.session_state['selected_rotation'] = 0
                if 'rotated_imgs' not in st.session_state:
                    st.session_state['rotated_imgs'] = None
                if st.session_state.recon['cest'] is None:
                    data_cest = load_study.recon_bart(cest_path, folder_path)
                    st.session_state.recon['cest'] = data_cest
                if st.session_state.recon['cest'] is not None:
                    if st.session_state.rot_done == False:
                        load_study.rotate_imgs(st.session_state)
                    elif st.session_state.rot_done == True:
                        st.success("Rotation finalized!")
                        if st.session_state.drift_done == False:
                            load_study.thermal_drift(st.session_state)
                        elif st.session_state.drift_done == True:
                            st.success("Thermal drift correction complete!")
                        if st.session_state.rois_done == False:
                            if st.session_state.submitted_data['organ'] == 'Cardiac':
                                draw_rois.cardiac_roi(st.session_state, st.session_state.recon['cest'])
                            if st.session_state.submitted_data['organ'] == 'Other':
                                draw_rois.draw_rois(st.session_state, st.session_state.recon['cest'])
                        elif st.session_state.rois_done == True:
                            st.success("ROIs submitted!")
                            st.write(st.session_state.user_geometry["rois"])
                                
                            

if st.button("Reset"):
    st.error("To reset and resubmit, please refresh the page.")