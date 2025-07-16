#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 12:35:44 2025

@author: jonah
"""
# --- Imports --- #
# Standard library imports
import os 
from pathlib import Path 
# Third-party imports
import streamlit as st
# Local application imports
from scripts import load_study, pre_processing, draw_rois, cest_fitting, quesp_fitting, plotting, plotting_quesp, plotting_wassr, plotting_damb1, BrukerMRI
from custom import st_functions

# --- Constants for app setup --- #
SITE_ICON = "./custom/icons/ksp.ico"
LOADING_GIF_PATH = Path("custom/icons/loading.gif")

# --- Session state management --- #
def initialize_session_state():
    """
    Initializes all necessary state condition variables with a checklist system.
    """
    defaults = {
        # Core app state
        "is_submitted": False,
        "processing_active": False,
        "is_processed": False,
        "display_data": False,
        # User selections
        "submitted_data": {},
        "custom_contrasts": None,
        "reference": None,
        # Checklist for pipeline stages
        "pipeline_status": {
            "recon_done": False,
            "orientation_done": False,
            "processing_done": False,
            "rois_done": False, # ROI drawing is a single event
            "fitting_done": [],
            },
        # Data storage
        "recon_data": {},
        "orientation_params": {"radial": None, "rectilinear": None},
        "processed_data": {},
        "user_geometry": {"rois": None, "masks": None, "aha": None},
        "fits": {},
        # Log messages
        "log_messages": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def clear_session_state():
    """
    Clears all keys from the session state.
    This is used to reset the app.
    """
    for key in list(st.session_state.keys()):
        del st.session_state[key]

# --- Data validation --- #
def validate_radial(path):
    """
    Check the existence of required files in a radial experiment.
    """
    required_files = ['method', 'acqp', 'traj', 'fid']
    missing = [file for file in required_files if not os.path.isfile(os.path.join(path, file))]
    return missing

def validate_rectilinear(path):
    """
    Check the existence of required files for rectilinear acquisition.
    """
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

def validate_double_angle(directory, theta_path, two_theta_path):
    """
    Check flip angles to make sure it's really double angle method.
    """
    exp_theta = BrukerMRI.ReadExperiment(directory, theta_path)
    exp_two_theta = BrukerMRI.ReadExperiment(directory, two_theta_path)
    theta = exp_theta.acqp['ACQ_flip_angle']
    two_theta = exp_two_theta.acqp['ACQ_flip_angle']
    if 2*theta != two_theta:
        return True, theta, two_theta
    elif 2*theta == two_theta:
        return False, theta, two_theta

def validate_fp_quesp(directory, quesp_path, t1_path):
    """
    Check to make sure the sequence is actually fp_EPI
    """ # Can remove when additional sequences are added
    exp_quesp = BrukerMRI.ReadExperiment(directory, quesp_path)
    exp_t1 = BrukerMRI.ReadExperiment(directory, t1_path)
    check_quesp = exp_quesp.method['Method']
    check_t1 = exp_t1.method['Method']
    if check_quesp != "<User:fp_EPI>" or check_t1 != "<Bruker:RAREVTR>":
        return True, check_quesp, check_t1 
    else:
        return False, check_quesp, check_t1

# --- UI functions --- #
def render_sidebar():
    """
    Renders sidebar content (disclaimers, contact info, etc.)
    """
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

def do_data_submission():
    """
    Handles the data submission form.
    """
    options = ["CEST", "QUESP", "WASSR", "DAMB1"]
    organs = ["Cardiac", "Other"]
    col1, col2 = st.columns((1,1))
    with col1:
        selection = st.multiselect("Experiment type(s)", options)
    with col2:
        anatomy = st.pills("ROI", organs)
    
    if selection and anatomy:
        folder_path = st.text_input('Input data path', placeholder='User/Documents/MRI_Data/Project/Scan_ID')
        save_path = st.text_input('Save processed data as', placeholder='Liver_5uT_Radial')
        
        cest_validation = True
        quesp_validation = True
        wassr_validation = True
        damb1_validation = True
        all_fields_filled = True  

        if not folder_path or not save_path:
            all_fields_filled = False
    
        if folder_path and os.path.isdir(folder_path):
            
            # CEST validation
            if "CEST" in selection:
                cest_path = st.text_input('Input CEST experiment number', placeholder='5')
                if not cest_path:
                    all_fields_filled = False  # CEST path is required
                if cest_path:
                    smoothing_filter = True
                    moco_cest = False
                    pca = False
                    pixelwise = False
                    cest_type = st.radio('CEST acquisition type', ["Radial", "Rectilinear"], horizontal=True)
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

            # QUESP validation
            if "QUESP" in selection:
                if anatomy == 'Cardiac':
                    quesp_validation = False
                    st.error("QUESP analysis is only supported for non-cardiac ROIs at this time.")
                else:
                    quesp_path = st.text_input('Input QUESP experiment number', help="Currently, only QUESP experiments run using the 'fp_EPI' sequence are supported.")
                    t1_path = st.text_input('Input T1 mapping experiment number', help="Currently, only VTR RARE T1 mapping is supported.")
                    if quesp_path and t1_path:
                        quesp_type = st.radio('QUESP analysis type', ["Standard (MTRasym)", "Inverse (MTRrex)"], horizontal=True)
                        if not quesp_type:
                            all_fields_filled = False
                        quesp_full_path = os.path.join(folder_path, quesp_path)
                        t1_full_path = os.path.join(folder_path, t1_path)
                        quesp_folder_exists = os.path.isdir(quesp_full_path)
                        t1_folder_exists = os.path.isdir(t1_full_path)
                        if not quesp_folder_exists:
                            st.error(f"QUESP folder does not exist: {quesp_full_path}")
                            quesp_validation = False
                        if not t1_folder_exists:
                            st.error(f"T1 map folder does not exist: {t1_full_path}")
                            quesp_validation = False
                        if quesp_folder_exists and t1_folder_exists:
                            st.success("QUESP and T1 map folders found!")
                            bad_method, check_quesp, check_t1 = validate_fp_quesp(folder_path, quesp_path, t1_path)
                            if bad_method:
                                quesp_validation = False
                                if check_quesp != "<User:fp_EPI>":
                                    st.error(f"Incorrect QUESP method detected: **{check_quesp}**. Only **<User:fp_EPI>** is supported.")
                                if check_t1 != "<Bruker:RAREVTR>":
                                    st.error(f"Incorrect T1 mapping method detected: **{check_t1}**. Only **<Bruker:RAREVTR>** is supported.")
                            else:
                                st.success("Method validation successful!")
                    else:
                         all_fields_filled = False

            # WASSR validation
            if "WASSR" in selection:
                wassr_path = st.text_input('Input WASSR experiment number')
                if not wassr_path:
                    all_fields_filled = False  # WASSR path is required
                if wassr_path:
                    moco_wassr = False
                    wassr_type = st.radio('WASSR acquisition type', ["Radial", "Rectilinear"], horizontal=True)
                    full_b0_mapping = st.toggle('Full B0 mapping', value=False, help="Fit B0 map for the entire image. Slower, but allows for full map visualization.") 
                    if "WASSR" in selection and wassr_type == "Radial":
                        moco_wassr = st.toggle('Motion correction (WASSR)', help="Correct bulk motion by discarding spokes based on projection images.")
                    if not wassr_type:
                        all_fields_filled = False  
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
                    all_fields_filled = False 
    
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

                    if os.path.isdir(theta_full_path) and os.path.isdir(two_theta_full_path):
                        bad_flips, theta, two_theta = validate_double_angle(folder_path, theta_path, two_theta_path)  
                        if bad_flips:
                            st.error("Incorrect flip angles: α = %i, 2α = %i" % (theta, two_theta))
                        else:
                            st.success("Flip angle validation successful! ")
            
            # Check if all fields are filled before enabling submit
            if all_fields_filled and (cest_validation and wassr_validation and damb1_validation and quesp_validation):
                if 'reference' in locals() and reference and reference_validation == False:
                    st.error("Please validate the additional reference image before submitting.")
                else:
                    if st.button("Submit"):
                        st.session_state.is_submitted = True
                        st.session_state.processing_active = True
                        # Ensure Data folder exists within the main path
                        data_folder = os.path.join(folder_path, "Data")
                        if not os.path.isdir(data_folder):
                            os.makedirs(data_folder)
                        
                        # Create folder for save_path within Data
                        save_full_path = os.path.join(data_folder, save_path)
                        if not os.path.isdir(save_full_path):
                            os.makedirs(save_full_path)
                        save_path = save_full_path  # Overwrite save_path with the full path
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
                        if "QUESP" in selection:
                            st.session_state.submitted_data['quesp_path'] = quesp_path
                            st.session_state.submitted_data['t1_path'] = t1_path
                            st.session_state.submitted_data['quesp_type'] = quesp_type
                        st.rerun()
            else:
                if not all_fields_filled:
                    st.error("Please fill in all the required fields before submitting.")
        else:
            if folder_path:
                st.error(f"The provided data path does not exist: {folder_path}")

def do_processing_pipeline():
    """Manages the sequential processing pipeline for all experiment types."""
    # Retrieve submitted experiment types 
    submitted = st.session_state.submitted_data
    selection = [s.lower() for s in submitted.get('selection', [])]
    # --- Stage 1: Reconstruction --- #
    # Reconstruct all selected data types if they haven't been already.
    if not st.session_state.pipeline_status.get('recon_done', False):
        with st.spinner("Reconstructing data..."):
            tasks_to_run = [exp for exp in selection if exp not in st.session_state.recon_data]
            for exp_type in tasks_to_run:
                if exp_type == 'cest':
                    cest_type = submitted.get('cest_type')
                    if cest_type == 'Radial' and submitted.get('moco_cest'):
                        st.session_state.recon_data['cest'] = pre_processing.run_radial_preprocessing(
                            submitted['folder_path'],
                            submitted['cest_path'],
                            submitted.get('pca', False),
                            exp_type
                        )
                    elif cest_type == 'Radial':
                        st.session_state.recon_data['cest'] = load_study.recon_bart(
                            submitted['cest_path'], submitted['folder_path']
                        )
                    else: # Rectilinear
                        st.session_state.recon_data['cest'] = load_study.recon_bruker(
                            submitted['cest_path'], submitted['folder_path']
                        )
                if exp_type == 'wassr':
                    wassr_type = submitted.get('wassr_type')
                    if wassr_type == 'Radial' and submitted.get('moco_wassr'):
                        st.session_state.recon_data['wassr'] = pre_processing.run_radial_preprocessing(
                            submitted['folder_path'],
                            submitted['wassr_path'],
                            False,
                            exp_type 
                        )
                    elif wassr_type == 'Radial':
                        st.session_state.recon_data['wassr'] = load_study.recon_bart(
                            submitted['wassr_path'], submitted['folder_path']
                        )
                    else: # Rectilinear
                        st.session_state.recon_data['wassr'] = load_study.recon_bruker(
                            submitted['wassr_path'], submitted['folder_path']
                        )
                if exp_type == "damb1":
                    st.session_state.recon_data['damb1'] = load_study.recon_damb1(submitted['folder_path'], submitted['theta_path'], submitted['two_theta_path'])
                if exp_type == "quesp":
                    st.session_state.recon_data['quesp'] = load_study.recon_quesp(submitted['quesp_path'], submitted['folder_path'])
                    st.session_state.recon_data['t1'] = load_study.recon_t1map(submitted['t1_path'], submitted['folder_path'])
            st.session_state.pipeline_status['recon_done'] = True
            st_functions.message_logging("All reconstruction complete!")
            st.rerun()
    
    # --- Stage 2: Group experiments and orient each group --- #
    if st.session_state.pipeline_status.get('recon_done') and not st.session_state.pipeline_status.get('orientation_done', False):
        # Group experiments by their trajectory type
        radial_exps = [exp for exp in ['cest', 'wassr'] if exp in st.session_state.recon_data and submitted.get(f'{exp}_type') == 'Radial']
        rectilinear_exps = [exp for exp in ['cest', 'wassr'] if exp in st.session_state.recon_data and submitted.get(f'{exp}_type') == 'Rectilinear']
        if 'damb1' in st.session_state.recon_data:
            rectilinear_exps.append('damb1')
        if 'quesp' in st.session_state.recon_data:
            rectilinear_exps.append('quesp')
        # Orient radial group
        if radial_exps and st.session_state.orientation_params.get('radial') is None:
            primary_exp = radial_exps[0] # Orient using the first radial experiment
            transforms = load_study.show_rotation_ui(st.session_state.recon_data[primary_exp]['imgs'], 'Radial')
            if transforms:
                st.session_state.orientation_params['radial'] = transforms
                st.rerun()
            else:
                return
        # Orient rectilinear group
        if rectilinear_exps and st.session_state.orientation_params.get('rectilinear') is None:
            primary_exp = rectilinear_exps[0] # Orient using the first rectilinear experiment
            transforms = load_study.show_rotation_ui(st.session_state.recon_data[primary_exp]['imgs'], 'Rectilinear')
            if transforms:
                st.session_state.orientation_params['rectilinear'] = transforms
                st.rerun()
            else:
                return
        # Check for completion
        radial_done = not radial_exps or st.session_state.orientation_params.get('radial') is not None
        rectilinear_done = not rectilinear_exps or st.session_state.orientation_params.get('rectilinear') is not None
        if radial_done and rectilinear_done:
            st.session_state.pipeline_status['orientation_done'] = True
            st_functions.message_logging("All orientations finalized!")
            st.rerun()

    # --- Stage 3: Apply transformations and corrections --- #
    if st.session_state.pipeline_status.get('orientation_done') and not st.session_state.pipeline_status.get('processing_done', False):
        with st.spinner("Applying orientation and corrections..."):
            for exp_type in selection:
                if exp_type in selection:
                    # Determine which orientation params to use
                    orientation_type = 'rectilinear' if exp_type in ['damb1', 'quesp'] or submitted.get(f'{exp_type}_type') == 'Rectilinear' else 'radial'
                    k, flip = st.session_state.orientation_params[orientation_type]
                    recon = st.session_state.recon_data[exp_type]
                    oriented = load_study.rotate_image_stack(recon['imgs'], k)
                    if flip:
                        oriented = load_study.flip_image_stack_vertically(oriented)
                    # Apply further corrections
                    if 'offsets' in recon and 'powers' not in recon: # CEST/WASSR
                        corrected = load_study.thermal_drift({"imgs": oriented, "offsets": recon['offsets']})
                        st.session_state.processed_data[exp_type] = corrected
                    elif 'powers' in recon: # QUESP
                        corrected = load_study.process_quesp({"imgs": oriented, "powers": recon['powers'], "times": recon['times'], "offsets": recon['offsets']})
                        st.session_state.processed_data[exp_type] = corrected
                    else: # DAMB1
                        st.session_state.processed_data[exp_type] = {"imgs": oriented, "nominal_flip": recon['nominal_flip']}
            st.session_state.pipeline_status['processing_done'] = True
            st_functions.message_logging("All data transformed and corrected!")
            st.rerun()

    # --- Stage 4: ROI drawing --- #
    if st.session_state.pipeline_status.get('processing_done') and not st.session_state.pipeline_status.get('rois_done', False):
        roi_canvas_placeholder = st.empty()
        with roi_canvas_placeholder.container():
            # Determine the best reference image for drawing ROIs
            ref_img = None
            if submitted.get('reference'):
                ref_img = st.session_state.submitted_data['reference']
            else:
                primary_exp = selection[0]
                processed_exp_data = st.session_state.processed_data[primary_exp]
                if 'm0' in processed_exp_data:
                    ref_image = processed_exp_data['m0']
                elif 'imgs' in processed_exp_data:
                    img_stack = processed_exp_data['imgs']
                    ref_image = img_stack[:, :, 0] if img_stack.ndim >= 3 else img_stack
            rois = draw_rois.cardiac_roi(ref_image) if submitted['organ'] == 'Cardiac' else draw_rois.draw_rois(ref_image)
        if rois:
            st.session_state.user_geometry['rois'] = rois
            st.session_state.pipeline_status['rois_done'] = True
            st_functions.message_logging("ROI definition complete!")
            roi_canvas_placeholder.empty()
            st.rerun()
        else:
            return

    # --- Stage 5: Fitting --- #
    if st.session_state.pipeline_status.get('rois_done') and not st.session_state.pipeline_status.get('fitting_done', False):
        with st.spinner("Performing final analysis..."):
            # --- Generate masks and AHA segments ---
            # Determine reference image
            ref_img = None
            if submitted.get('reference'):
                ref_img = st.session_state.submitted_data['reference']
            else:
                primary_exp = selection[0]
                processed_exp_data = st.session_state.processed_data[primary_exp]
                if 'm0' in processed_exp_data:
                    ref_image = processed_exp_data['m0']
                elif 'imgs' in processed_exp_data:
                    img_stack = processed_exp_data['imgs']
                    ref_image = img_stack[:, :, 0] if img_stack.ndim >= 3 else img_stack
            masks = draw_rois.convert_rois_to_masks(ref_image, st.session_state.user_geometry['rois'])
            st.session_state.user_geometry['masks'] = masks
            
            if submitted['organ'] == 'Cardiac':
                lv_mask = draw_rois.calc_lv_mask(masks)
                st.session_state.user_geometry['masks']['lv'] = lv_mask
                st.session_state.user_geometry['aha'] = draw_rois.aha_segmentation(lv_mask, masks['insertion_points'])

            # --- Run fitting for all selected types --- #
            if "cest" in selection:
                proc_data = st.session_state.processed_data['cest']
                spectra = cest_fitting.calc_spectra(proc_data['imgs'], st.session_state.user_geometry)
                st.session_state.fits['cest'] = cest_fitting.fit_all_rois(spectra, proc_data['offsets'], submitted.get('custom_contrasts'))
                if submitted.get('pixelwise'):
                    pixel_spectra = cest_fitting.calc_spectra_pixelwise(proc_data['imgs'], st.session_state.user_geometry['masks'])
                    st.session_state.fits['cest_pixelwise'] = cest_fitting.fit_all_pixels(pixel_spectra, proc_data['offsets'], submitted.get('custom_contrasts'))
                if submitted['organ'] == 'Cardiac':
                    cest_fits = st.session_state.fits.get('cest', {})
                    segments_to_check = ["Anterior", "Anteroseptal"] # Can be changed if needed
                    for segment in segments_to_check:
                        fit_data = cest_fits.get(segment)
                        if fit_data:
                            rmse = fit_data.get("RMSE")
                            if rmse is not None and rmse > 0.02:
                                st_functions.message_logging(f"Fit RMSE in {segment.lower()} segment > 2% (RMSE = {rmse*100:.3f}%)!", msg_type='warning')

            if "quesp" in selection:
                t1_fits = quesp_fitting.fit_t1_map(st.session_state.recon_data['t1'], masks)
                st.session_state.fits['t1'] = t1_fits
                st.session_state.fits['quesp'] = quesp_fitting.fit_quesp_map(st.session_state.processed_data['quesp'], t1_fits, masks, submitted.get('quesp_type'))
            
            if "wassr" in selection:
                proc_data = st.session_state.processed_data['wassr']
                if submitted.get('full_b0_mapping'):
                    st.session_state.fits['wassr'], st.session_state.fits['wassr_full_map'] = cest_fitting.fit_wassr_full(proc_data['imgs'], proc_data['offsets'], st.session_state.user_geometry)
                else:
                    st.session_state.fits['wassr'] = cest_fitting.fit_wassr_masked(proc_data['imgs'], proc_data['offsets'], st.session_state.user_geometry)

            if "damb1" in selection:
                proc_data = st.session_state.processed_data['damb1']
                st.session_state.fits['damb1'] = cest_fitting.fit_b1(proc_data['imgs'], proc_data['nominal_flip'])

            st_functions.message_logging("All processing complete!")
            st.session_state.pipeline_status['fitting_done'] = True
            st.session_state.is_processed = True
            st.session_state.display_data = True
            st.session_state.processing_active = False
            st.rerun()

def display_results():
    """
    Displays all the final plots and data.
    """
    submitted = st.session_state.submitted_data
    save_path = submitted['save_path']
    
    if "CEST" in submitted['selection']:
        st.header('CEST Results')
        ref_image = st.session_state.processed_data['cest']['m0']
        mask = st.session_state.user_geometry['masks']['lv']
        if submitted['organ'] == 'Cardiac':
            plotting.show_segmentation(ref_image, mask, st.session_state.user_geometry['aha'], save_path)
        else:
            plotting.show_rois(ref_image, st.session_state.user_geometry['masks'], save_path)
        
        if submitted.get('pixelwise') and 'cest_pixelwise' in st.session_state.fits:
            plotting.pixelwise_mapping(
                ref_image, st.session_state.fits['cest_pixelwise'], 
                st.session_state.user_geometry,
                submitted.get('custom_contrasts'), submitted.get('smoothing_filter'), save_path
            )

        plotting.plot_zspec(st.session_state.fits['cest'], save_path)
        
    if "QUESP" in submitted['selection']:
        st.header('QUESP Results')
        col1, col2 = st.columns(2)
        with col1:
            plotting_quesp.plot_t1_map(st.session_state.fits['t1'], st.session_state.processed_data['quesp']['m0'], st.session_state.user_geometry['masks'], save_path)
        with col2:
            plotting.show_rois(st.session_state.processed_data['quesp']['m0'], st.session_state.user_geometry['masks'], save_path)
        plotting_quesp.plot_quesp_maps(st.session_state.fits['quesp'], st.session_state.user_geometry['masks'], st.session_state.processed_data['quesp']['m0'], save_path)
        st.subheader("Statistics")
        stats_df = plotting_quesp.calculate_quesp_stats(st.session_state.fits['quesp'], st.session_state.fits['t1'])
        st.dataframe(stats_df.style.format("{:.4f}"))
        st_functions.save_df_to_csv(stats_df, save_path)
        st.warning('Plot colorbars and statistics are displayed within the 5-95th percentile range per ROI.')

    if "WASSR" in submitted['selection']:
        st.header('WASSR Results')
        ref_image = st.session_state.processed_data['cest']['m0'] if 'cest' in st.session_state.processed_data else st.session_state.processed_data['wassr']['m0']
        plotting_wassr.plot_wassr(ref_image, st.session_state.user_geometry, st.session_state.fits.get('wassr'), save_path,st.session_state.fits.get('wassr_full_map'))
        if submitted['organ'] == 'Cardiac':
            plotting_wassr.plot_wassr_aha(st.session_state.fits['wassr'], save_path)

    if "DAMB1" in submitted['selection']:
        st.header('DAMB1 Results')
        ref_image = st.session_state.processed_data['cest']['m0'] if 'cest' in st.session_state.processed_data else st.session_state.processed_data['wassr']['m0'] if 'wassr' in st.session_state.processed_data else None
        plotting_damb1.plot_damb1(st.session_state.fits['damb1'], ref_image, st.session_state.user_geometry, save_path)
        if submitted['organ'] == 'Cardiac':
            plotting_damb1.plot_damb1_aha(st.session_state.fits['damb1'], ref_image, st.session_state.user_geometry['aha'], save_path)

    st_functions.save_raw(st.session_state)
    if any(msg_type in ['warning', 'error'] for _, msg_type in st.session_state.log_messages):
        st.error("**One or more issues were noted during processing. Please review the log in the 'Process data' expander.**")
    st.success(f"Images, plots, and raw data saved at **{save_path}**")

# --- Main app --- #
def main():
    """
    Main function to run the Streamlit app.
    """
    # Setup
    st.set_page_config(page_title="Pre-CAT", initial_sidebar_state="expanded", page_icon=SITE_ICON)
    if LOADING_GIF_PATH.exists():
        st_functions.inject_custom_loader(LOADING_GIF_PATH)
    st_functions.inject_spinning_logo_css(SITE_ICON)
    initialize_session_state()
    render_sidebar()
    hoverable_pre_cat = st_functions.add_hoverable_title_with_image_inline(
        "Pre-CAT", "https://i.ibb.co/gMQ7MCb/Subject-4.png"
    )
    st.markdown(
        f"<h1 style='font-size: 3rem; font-weight: bold;'>Welcome to {hoverable_pre_cat}</h1>",
        unsafe_allow_html=True
    )
    st.write("### A preclinical CEST-MRI analysis toolbox.")
    # Main state machine
    with st.expander("Load data", expanded=not st.session_state.is_submitted):
        do_data_submission()
    if st.session_state.is_submitted:
        with st.expander("Process data", expanded=st.session_state.processing_active):
            for msg, msg_type in st.session_state.get("log_messages", []):
                if msg_type == 'success':
                    st.success(msg)
                elif msg_type == 'warning':
                    st.warning(msg)
                elif msg_type == 'error':
                    st.error(msg)
                else:
                    st.info(msg)
            do_processing_pipeline()
    if st.session_state.is_processed:
        with st.expander("Display and save results", expanded=st.session_state.display_data):
            display_results()

    if st.button("Reset"):
        clear_session_state()
        st.rerun()

if __name__ == "__main__":
    main()