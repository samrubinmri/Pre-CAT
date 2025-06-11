#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 12:35:44 2025

@author: jonah
"""

import streamlit as st
import os
from scripts import load_study, pre_processing, draw_rois, cest_fitting, plotting, plotting_wassr, plotting_damb1
from custom import st_functions
from pathlib import Path

site_icon = "./custom/icons/ksp.ico"
loading_gif_path = Path("custom/icons/loading.gif")
st.set_page_config(page_title="Pre-CAT", initial_sidebar_state="expanded", page_icon = site_icon)
if loading_gif_path.exists():
    st_functions.inject_custom_loader(loading_gif_path)

if "is_submitted" not in st.session_state:
    st.session_state.is_submitted = False
if "submitted_data" not in st.session_state:
    st.session_state.submitted_data = {}
if "processing_active" not in st.session_state:
    st.session_state.processing_active = False
if "is_processed" not in st.session_state:
    st.session_state.is_processed = False
if "display_data" not in st.session_state:
    st.session_state.display_data = False
if "custom_contrasts" not in st.session_state:
    st.session_state.custom_contrasts = None
if "reference" not in st.session_state:
    st.session_state.reference = None
    
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
    "https://i.ibb.co/gMQ7MCb/Subject-4.png"  # Replace with your image URL
)

# Combine the static title and hoverable title into one header
st.markdown(
    f"<h1 style='font-size: 3rem; font-weight: bold;'>Welcome to {hoverable_pre_cat}</h1>",
    unsafe_allow_html=True
)

# Add description text
st.write("### A preclinical CEST-MRI analysis toolbox.")
with st.sidebar:
    st.write("""## Instructions and Disclaimer
Specify experiment type(s), ROI, and file locations for raw data.

Follow each subsequent step after carefully reading associated instructions.

**For users unfamiliar with cardiac anatomy and terminology, detailed instructions for ROI prescription are included in the [Github repository](https://github.com/jweigandwhittier/Pre-CAT/blob/main/instructions/cardiac_rois.pdf).**

When using **Pre-CAT**, please remember the following:
- **Pre-CAT** is not licensed for clinical use and is intended for research purposes only.
- Due to B0 inhomogeneities, cardiac CEST data is only useful in anterior segments.
- Each raw data file includes calculated RMSE in the CEST fitting region. Please refer to this if output data seem noisy.
- By default, **Pre-CAT** fits two rNOE peaks at frequency offsets -1.6 ppm (upper bound: -1.2 ppm, lower bound: -1.8 ppm) and -3.5 ppm (upper bound: -3.2 ppm, lower bound: -4.0 ppm) per *Zhang et al. Magnetic Resonance Imaging, Oct. 2016, doi: 10.1016/j.mri.2016.05.002*.
             
    """)
    st.write("""## Citation
This webapp is associated with the following paper, please cite this work when using **Pre-CAT**. \n
Weigand-Whittier J, Wendland M, Lam B, et al. *Ungated, plug-and-play cardiac CEST-MRI using radial FLASH with segmented saturation*. Magn Reson Med (2024). 10.1002/mrm.30382""")

    st_functions.inject_hover_email_css()
    
    st.write("## Contact")
    
    st.markdown(f"""
    <p style="margin-bottom: 0">
    Contact me with any issues or questions: 
    <span class="hoverable-email">
        <a href="mailto:jweigandwhittier@berkeley.edu">jweigandwhittier@berkeley.edu</a>
        <span class="image-tooltip">
            <img src="https://i.ibb.co/M5h9MyF1/Subject-5.png" alt="Hover image">
        </span>
    </span>
    </p>
    <br>
    """, unsafe_allow_html=True)
    
    st.write("Please add **[Pre-CAT]** to the subject line of your email.")

with st.expander("Load data", expanded = not st.session_state.is_submitted):
    options = ["CEST", "WASSR", "DAMB1"]
    organs = ["Cardiac", "Other"]
    col1, col2 = st.columns((1,1))
    with col1:
        selection = st.pills("Experiment type(s)", options, selection_mode="multi")
    with col2:
        anatomy = st.pills("ROI", organs)
    
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
                    #col1, col2 = st.columns(2)
                    #with col1:
                    cest_type = st.radio('CEST acquisition type', ["Radial", "Rectilinear"], horizontal=True)
                    #with col2:
                    st.markdown(
                    """
                    <style>
                    .custom-label {
                        font-size: 0.875rem; /* Matches theme.fontSizes.sm */
                        display: flex;
                        align-items: center;
                        margin-bottom: 0.25rem; /* Matches theme.spacing.twoXS */
                        min-height: 1.25rem; /* Matches theme.fontSizes.xl */
                        font-family: 'Source Sans Pro', sans-serif;
                        font-weight: normal; /* Ensure weight matches */
                        line-height: 1.6; /* Ensures vertical alignment */
                    }
                    </style>
                    <label class="custom-label">
                      Additional settings
                    </label>
                    </div>
                    """,
                    unsafe_allow_html=True,
                    )
                    if "CEST" in selection and cest_type == "Radial":
                        moco_cest = st.toggle('Motion correction (CEST)', help="Correct bulk motion by discarding spokes based on projection images.")
                        if moco_cest:
                            pca = st.toggle('Z-spectral denoising', help="Z-spectral denoising with principal component analysis. This is a *global* method using Malinowskis empirical indicator function.")
                    pixelwise = st.toggle(
                        'Pixelwise mapping', help="Accuracy is highly dependent on field homogeneity.")
                    if pixelwise:
                        smoothing_filter = st.toggle('Median smoothing filter', help="Apply a median filter to smooth contrast maps.")
                    if anatomy == "Other":
                        reference = st.toggle(
                            'Additional reference image', help="Use this option to load an additional reference image for ROI(s)/masking. By default, the unsaturated (S0/M0) image is used.")
                        if reference:
                            all_fields_filled = False
                            reference_path = st.text_input('Input reference experiment number', help='Reference image assumed to be rectilinear. Please only use single slice images.')
                            if reference_path:
                                reference_full_path = os.path.join(folder_path, reference_path)
                                all_fields_filled = True
                                reference_validation = False
                                if os.path.isdir(reference_full_path):  
                                    reference_validation = True
                                    missing_items = validate_rectilinear(reference_full_path)
                                    if missing_items:
                                        st.error(f"Reference folder is missing the following required items: {', '.join(missing_items)}")
                                        reference_validation = False
                                    else:
                                        reference_image = load_study.load_bruker_img(reference_path, folder_path)
                                        if reference_image.shape[2] != 1:
                                            st.error("Reference image contains multislice data! Currently, only single slice data is allowed.")
                                            reference_validation = False
                                        else:
                                           st.session_state.reference = reference_image 
                                else:
                                    st.error(f"Reference folder does not exist: {reference_full_path}")
                                    reference_validation = False
                                
                        choose_contrasts = st.toggle(
                            'Choose contrasts', help="Default contrasts are: amide, creatine, NOE. Water and MT are always fit.")
                        if choose_contrasts:
                            contrasts = ["NOE (-3.5 ppm)", "Amide", "Creatine", "Amine", "Hydroxyl", "NOE (-1.6 ppm)"]
                            default_contrasts = ["NOE (-3.5 ppm)", "Amide", "Creatine", "NOE (-1.6 ppm)"]
                            contrast_selection = st.pills ("Contrasts", contrasts, default=default_contrasts, selection_mode="multi")
                            st.session_state.custom_contrasts = contrast_selection
                        else:
                            st.session_state.custom_contrasts = None
                    elif anatomy == "Cardiac":
                        st.session_state.reference = None
                        st.session_state.custom_contrasts = None
                    if not cest_type:
                        all_fields_filled = False  
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
                    full_b0_mapping = st.toggle('Full B0 mapping', value=True, help="Fit B0 map for the entire image. Slower, but allows for full map visualization.") 
                    if "WASSR" in selection and wassr_type == "Radial":
                        moco_wassr = st.toggle('Motion correction (WASSR)', help="Correct bulk motion by discarding spokes based on projection images.")
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
                if 'reference' in locals() and reference and reference_validation == False:
                    st.error("Please validate the additional reference image before submitting.")
                else:
                    if st.button("Submit"):
                        st.session_state.is_submitted = True
                        st.session_state.submitted_data = {
                            "folder_path": folder_path,
                            "save_path": save_path,
                            "selection": selection,
                            "organ": anatomy}
                        if "CEST" in selection:
                            st.session_state.submitted_data['cest_path'] = cest_path
                            st.session_state.submitted_data['cest_type'] = cest_type
                            st.session_state.submitted_data['pixelwise'] = pixelwise
                            st.session_state.submitted_data['smoothing_filter'] = smoothing_filter
                            st.session_state.submitted_data['moco_cest'] = moco_cest
                            st.session_state.submitted_data['pca'] = pca
                        if "WASSR" in selection: 
                            st.session_state.submitted_data['wassr_path'] = wassr_path
                            st.session_state.submitted_data['wassr_type'] = wassr_type
                            st.session_state.submitted_data['full_b0_mapping'] = full_b0_mapping
                            st.session_state.submitted_data['moco_wassr'] = moco_wassr
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
    with st.expander("Process data", expanded = not st.session_state.is_processed):
        # Set new session vars
        if 'cest_rot_exists' not in st.session_state:
            st.session_state['cest_rot_exists'] = False
        if "recon" not in st.session_state:
            st.session_state.recon = {}
            if 'CEST' in st.session_state.submitted_data['selection']:
                st.session_state.recon["cest"] = None
            if 'WASSR' in st.session_state.submitted_data['selection']:
                st.session_state.recon["wassr"] = None
            if 'DAMB1' in st.session_state.submitted_data['selection']:
                st.session_state.recon["damb1"] = None
        if "user_geometry" not in st.session_state:
            st.session_state.user_geometry = {
                "rotations": None,
                "rois": None,
                "masks": None}
            if st.session_state.submitted_data['organ'] == 'Cardiac':
                st.session_state.user_geometry["aha"] = None
        if "processed_data" not in st.session_state:
            st.session_state.processed_data = {}
            if 'CEST' in st.session_state.submitted_data['selection']:
                st.session_state.processed_data["spectra"] = None
                st.session_state.processed_data["fits"] = None
                if st.session_state.submitted_data['pixelwise'] == True:
                    st.session_state.processed_data["pixelwise"] = {
                        "spectra":None,
                        "fits":None,
                        "maps":None}
            if 'WASSR' in st.session_state.submitted_data['selection']:
                st.session_state.processed_data["wassr_fits"] = None
                st.session_state.processed_data["wassr_full_map"] = None
            if 'DAMB1' in st.session_state.submitted_data['selection']:
                st.session_state.processed_data["b1_fits"] = None
        if "loading_done" not in st.session_state:
            st.session_state.loading_done = {}
            if 'CEST' in st.session_state.submitted_data['selection']:
                st.session_state.loading_done["cest"] = False
            if 'WASSR' in st.session_state.submitted_data['selection']:
                st.session_state.loading_done["wassr"] = False
            if 'DAMB1' in st.session_state.submitted_data['selection']:
                st.session_state.loading_done["damb1"] = False
        if "rot_done" not in st.session_state:
            st.session_state.rot_done = False
        if "drift_done" not in st.session_state:
            st.session_state.drift_done = {}
            if 'CEST' in st.session_state.submitted_data['selection']:
                st.session_state.drift_done["cest"] = False
            if 'WASSR' in st.session_state.submitted_data['selection']:
                st.session_state.drift_done["wassr"] = False
            if 'DAMB1' in st.session_state.submitted_data['selection']:
                st.session_state.drift_done["damb1"] = False
        if "rois_done" not in st.session_state:
            st.session_state.rois_done = False
        # Retrieve submitted data
        submitted_data = st.session_state.submitted_data
        ##--Logic for each experiment type--##
        ## CEST processing
        if "CEST" in submitted_data["selection"]:
            cest_path = submitted_data.get("cest_path")  # Retrieve cest_path from submitted data
            cest_type = submitted_data.get("cest_type")  # Retrieve cest_type from submitted data
            folder_path = submitted_data["folder_path"]
            if cest_type == 'Rectilinear':
                if st.session_state.recon['cest'] is None:
                    data_cest = load_study.recon_bruker(cest_path, folder_path)
                    st.session_state.recon['cest'] = data_cest
                    st.session_state.loading_done['cest'] = True
                    ## Add ability to rotate rectilinear data
            elif cest_type == 'Radial':
                if st.session_state.recon['cest'] is None:
                    if st.session_state.submitted_data["moco_cest"] == False:
                        data_cest = load_study.recon_bart(cest_path, folder_path)
                        st.session_state.recon['cest'] = data_cest
                    elif st.session_state.submitted_data["moco_cest"] == True:
                        data_cest = pre_processing.pre_processing(st.session_state, 'cest')
                        st.session_state.recon['cest'] = data_cest
            if 'rotation_stage' not in st.session_state:
                    st.session_state['rotation_stage'] = 'select_rotation'  # Stages: 'select_rotation', 'confirm_rotation', 'finalized'
            if 'selected_rotation' not in st.session_state:
                st.session_state['selected_rotation'] = 0
            if 'rotated_imgs' not in st.session_state:
                st.session_state['rotated_imgs'] = None
            if st.session_state.recon['cest'] is not None:
                if st.session_state.rot_done == False:
                    load_study.rotate_imgs(st.session_state, 'cest')
                elif st.session_state.rot_done == True:
                    st.success("Rotation finalized!")
                    st.session_state.loading_done['cest'] = True
                    st.session_state['cest_rot_exists'] = True
            if st.session_state.loading_done['cest'] == True:
                if st.session_state.drift_done['cest'] == False:
                    load_study.thermal_drift(st.session_state, 'cest')
                elif st.session_state.drift_done['cest'] == True:
                    st.success("Thermal drift correction (CEST) complete!")
                if st.session_state.rois_done == False:
                    if st.session_state.reference is not None:
                        reference = st.session_state.reference
                    else:
                        reference = st.session_state.recon['cest']
                    if st.session_state.submitted_data['organ'] == 'Cardiac':
                        draw_rois.cardiac_roi(st.session_state, reference, st.session_state.recon['cest'])
                    if st.session_state.submitted_data['organ'] == 'Other':
                        draw_rois.draw_rois(st.session_state, reference, st.session_state.recon['cest'])
                elif st.session_state.rois_done == True:
                    st.success("ROIs submitted!")
                    image = st.session_state.recon['cest']['m0']
                    rois = st.session_state.user_geometry["rois"]
                    st.session_state.user_geometry['masks'] = draw_rois.convert_rois_to_masks(image, rois)
                    masks = st.session_state.user_geometry['masks']
                    if st.session_state.submitted_data['organ'] == 'Cardiac':
                        st.session_state.user_geometry['masks']['lv'] = draw_rois.calc_lv_mask(masks)
                        draw_rois.aha_segmentation(image, st.session_state)
                    imgs = st.session_state.recon['cest']['imgs']
                    cest_fitting.calc_spectra(imgs, st.session_state)
                    st.session_state.processed_data["fits"] = cest_fitting.two_step(st.session_state.processed_data['spectra'], st.session_state.recon['cest']['offsets'], st.session_state.custom_contrasts)
                    if st.session_state.submitted_data['pixelwise'] == True and st.session_state.processed_data['pixelwise']['fits'] is None:
                        cest_fitting.calc_spectra_pixelwise(imgs, st.session_state)
                        st.session_state.processed_data['pixelwise']['fits'] = cest_fitting.per_pixel(st.session_state)
                    st.success("Fitting complete (CEST)!")
        ## WASSR processing
        if "WASSR" in submitted_data["selection"] and (
        "CEST" not in submitted_data["selection"] or 
        (st.session_state.loading_done.get("cest") and 
        st.session_state.drift_done["cest"] and 
        st.session_state.rois_done and 
        st.session_state.processed_data.get("fits") is not None)):
            wassr_path = submitted_data.get("wassr_path")  
            wassr_type = submitted_data.get("wassr_type")  
            folder_path = submitted_data["folder_path"]
            if wassr_type == 'Rectilinear':
                if st.session_state.recon['wassr'] is None:
                    data_wassr = load_study.recon_bruker(wassr_path, folder_path)
                    st.session_state.recon['wassr'] = data_wassr
                    st.session_state.loading_done['wassr'] = True
            elif wassr_type == 'Radial':
                if st.session_state.recon['wassr'] is None:
                    if st.session_state.submitted_data["moco_wassr"] == False:
                        data_wassr = load_study.recon_bart(wassr_path, folder_path)
                        st.session_state.recon['wassr'] = data_wassr
                    elif st.session_state.submitted_data["moco_wassr"] == True:
                        data_wassr = pre_processing.pre_processing(st.session_state, 'wassr')
                        st.session_state.recon['wassr'] = data_wassr
            if 'rotation_stage' not in st.session_state:
                st.session_state['rotation_stage'] = 'select_rotation'
            if 'selected_rotation' not in st.session_state:
                st.session_state['selected_rotation'] = 0
            if 'rotated_imgs' not in st.session_state:
                st.session_state['rotated_imgs'] = None
            if 'wassr_rot_exists' not in st.session_state:
                st.session_state['wassr_rot_exists'] = False
            if st.session_state.recon['wassr'] is not None:
                if st.session_state['cest_rot_exists'] == False and st.session_state.rot_done == False:
                    load_study.rotate_imgs(st.session_state, 'wassr')
                elif st.session_state.rot_done == True and st.session_state['cest_rot_exists'] == False:
                    st.session_state['wassr_rot_exists'] = True
                    st.session_state.loading_done['wassr'] = True
                    st.success("Rotation finalized!")
                elif st.session_state['cest_rot_exists'] == True:
                    load_study.quick_rot(st.session_state, 'wassr')
                    st.session_state.loading_done['wassr'] = True
            if st.session_state.loading_done['wassr'] == True:
                if st.session_state.drift_done['wassr'] == False:
                    load_study.thermal_drift(st.session_state, 'wassr')
                elif st.session_state.drift_done['wassr'] == True:
                    st.success("Thermal drift correction (WASSR) complete!")
            if st.session_state.rois_done == False and st.session_state.drift_done['wassr'] == True:
                if st.session_state.reference is not None:
                    reference = st.session_state.reference
                else:
                    reference = st.session_state.recon['wassr']
                if st.session_state.submitted_data['organ'] == 'Cardiac':
                    draw_rois.cardiac_roi(st.session_state, reference, st.session_state.recon['wassr'])
                if st.session_state.submitted_data['organ'] == 'Other':
                    draw_rois.draw_rois(st.session_state, reference, st.session_state.recon['wassr'])
            elif st.session_state.rois_done == True:
                if st.session_state.user_geometry['masks'] is None:
                    image = st.session_state.recon['wassr']['m0']
                    rois = st.session_state.user_geometry["rois"]
                    st.session_state.user_geometry['masks'] = draw_rois.convert_rois_to_masks(image, rois)
                    masks = st.session_state.user_geometry['masks']
                    if st.session_state.submitted_data['organ'] == 'Cardiac':
                        st.session_state.user_geometry['masks']['lv'] = draw_rois.calc_lv_mask(masks)
                        draw_rois.aha_segmentation(image, st.session_state)
                # Conditional fitting based on the toggle
                if st.session_state.processed_data.get('wassr_fits') is None:
                    imgs = st.session_state.recon['wassr']['imgs']
                    if st.session_state.submitted_data.get('full_b0_mapping', True): # Default to True if not set
                        st.session_state.processed_data['wassr_fits'], st.session_state.processed_data['wassr_full_map'] = cest_fitting.fit_wassr_full(imgs, st.session_state)
                    else:
                        st.session_state.processed_data['wassr_fits'] = cest_fitting.fit_wassr_masked(imgs, st.session_state)
                    st.success("Fitting complete (WASSR)!")
        ## DAMB1 processing
        cest_selected = "CEST" in submitted_data["selection"]
        wassr_selected = "WASSR" in submitted_data["selection"]
        damb1_selected = "DAMB1" in submitted_data["selection"]
        cest_ready = (
            st.session_state.loading_done.get("cest") and
            st.session_state.drift_done.get("cest") and
            st.session_state.rois_done and
            st.session_state.processed_data.get("fits") is not None
        )
        wassr_ready = (
            st.session_state.drift_done.get("wassr") and
            st.session_state.processed_data.get("wassr_fits") is not None
        )
        if damb1_selected and (
            (cest_selected and wassr_selected and cest_ready and wassr_ready) or
            (cest_selected and not wassr_selected and cest_ready) or
            (not cest_selected and wassr_selected and wassr_ready) or
            (not cest_selected and not wassr_selected)
        ):
            if 'rotation_stage' not in st.session_state:
                st.session_state['rotation_stage'] = 'select_rotation'  # Stages: 'select_rotation', 'confirm_rotation', 'finalized'
            if 'selected_rotation' not in st.session_state:
                st.session_state['selected_rotation'] = 0
            if 'rotated_imgs' not in st.session_state:
                st.session_state['rotated_imgs'] = None
            if 'damb1_rot_exists' not in st.session_state:
                st.session_state['damb1_rot_exists'] = False  
            if st.session_state.recon['damb1'] is None:
                st.session_state.recon["damb1"] = load_study.recon_damb1(st.session_state)
                # The premature setting of loading_done was removed from here.
                if wassr_selected and (st.session_state['wassr_rot_exists'] == True and st.session_state.submitted_data['wassr_type'] == 'Rectilinear'):
                    load_study.quick_rot(st.session_state, 'damb1')
                    st.session_state.loading_done['damb1'] = True
                if cest_selected and (st.session_state['cest_rot_exists'] == True and st.session_state.submitted_data['cest_type'] == 'Rectilinear'): # Corrected typo here
                    load_study.quick_rot(st.session_state, 'damb1')
                    st.session_state.loading_done['damb1'] = True
                else:
                    st.session_state['rotation_stage'] = 'select_rotation'
                    st.session_state['selected_rotation'] = 0
                    st.session_state['rotated_imgs'] = None
                    st.session_state['rot_done'] = False
                    st.session_state['damb1_rot_exists'] = True
            if st.session_state.recon['damb1'] and st.session_state['damb1_rot_exists'] == True:
                load_study.rotate_imgs(st.session_state, 'damb1')
            if st.session_state['damb1_rot_exists'] == True and st.session_state.rot_done == True:
                st.session_state.loading_done['damb1'] = True
            if st.session_state.loading_done['damb1'] == True:
                imgs = st.session_state.recon["damb1"]["imgs"]
                st.session_state.processed_data['b1_fits'] = cest_fitting.fit_b1(imgs, st.session_state)

    # Final check: Only mark as processed if all selected types are done
    required_keys = []
    if 'CEST' in submitted_data['selection']:
        required_keys.extend([
            st.session_state.drift_done.get('cest', False),
            st.session_state.loading_done.get('cest', False),
            st.session_state.processed_data.get('fits') is not None,
        ])
        if submitted_data['pixelwise']:
            required_keys.append(st.session_state.processed_data['pixelwise']['fits'] is not None)

    if 'WASSR' in submitted_data['selection']:
        required_keys.extend([
            st.session_state.recon.get('wassr') is not None,
            st.session_state.processed_data.get('wassr_fits') is not None
        ])

    if 'DAMB1' in submitted_data['selection']:
        required_keys.append(
            st.session_state.processed_data.get('b1_fits') is not None
        )

    # If all conditions are True, finalize processing
    if all(required_keys):
        st.session_state.processing_active = False
        st.session_state.is_processed = True
        st.session_state.display_data = True
                    
if st.session_state.display_data == True:      
    save_path = st.session_state.submitted_data["save_path"]             
    with st.expander('Display and save results', expanded = st.session_state.display_data):
        if "CEST" in submitted_data["selection"]:
            st.header('CEST')
            image = st.session_state.recon['cest']['m0']
            # if st.session_state.submitted_data['organ'] == 'Cardiac':
                # plotting.show_segmentation(image, st.session_state)
            # elif st.session_state.submitted_data['organ'] == 'Other':
                # plotting.show_rois(image, st.session_state)
            if st.session_state.submitted_data['pixelwise'] == True:
                plotting.pixelwise_mapping(image, st.session_state)
            plotting.plot_zspec(st.session_state)
            st_functions.save_raw(st.session_state)
            if st.session_state.submitted_data['organ'] == 'Cardiac': 
                rmse = st.session_state.processed_data["fits"]["Anterior"]["RMSE"]
                st.info(f'**Anterior fit RMSE:** {rmse*100:.3f}%')  
                if rmse > 0.02:
                    st.error("High RMSE in anterior segment! Recommend examining and/or excluding this dataset!")
        if "WASSR" in submitted_data["selection"]:
            st.header('WASSR')
            if "CEST" not in submitted_data["selection"]:
                image = st.session_state.recon['wassr']['m0']
            plotting_wassr.plot_wassr(image, st.session_state)
            if st.session_state.submitted_data['organ'] == 'Cardiac': 
                plotting_wassr.plot_wassr_aha(st.session_state)
        if "DAMB1" in submitted_data["selection"]:
            st.header('DAMB1')
            plotting_damb1.plot_damb1(st.session_state)
            if st.session_state.submitted_data['organ'] == 'Cardiac' and 'CEST' or 'WASSR' in session_state.submitted_data['selection']:
                plotting_damb1.plot_damb1_aha(st.session_state)
        st.success("Images, plots, and raw data saved at **%s**" % save_path)

if st.button("Reset"):
    st.error("To reset and resubmit, please refresh the page.")