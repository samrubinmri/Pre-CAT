#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 11:08:05 2024

@author: jonah
"""
import streamlit as st
import numpy as np
from sklearn.metrics import mean_squared_error
from scipy.optimize import curve_fit
from scipy.interpolate import CubicSpline

###Pre-correction###
##Starting points for curve fitting: amplitude, FWHM, peak center##
p0_water = [0.8, 1.8, 0]
p0_mt = [0.15, 40, -1]
##Lower bounds for curve fitting##
lb_water = [0.02, 0.3, -10]
lb_mt = [0.0, 30, -2.5]
##Upper bounds for curve fitting##
ub_water = [1, 10, 10]
ub_mt = [0.5, 60, 0]

##Try different starting points for water and creating FWHM in phantom##
# p0_water = [0.8, 0.1, 0]

##Combine for curve fitting##
#B0 correction (tissue)
p0_corr = p0_water + p0_mt
lb_corr = lb_water + lb_mt
ub_corr = ub_water + ub_mt 

#B0 correction (phantom)
p0_corr_ph = p0_water
lb_corr_ph = lb_water
ub_corr_ph = ub_water

###Post-correction###
##Starting points for curve fitting: amplitude, FWHM, peak center##
p0_water = [0.8, 0.2, 0]
p0_mt = [0.15, 40, -1]
p0_noe = [0.05, 1, -2.75]
p0_noe_neg_1_6 = [0.05, 1, -1.6]
p0_creatine = [0.05, 0.5, 2.0]
p0_amide = [0.05, 1.5, 3.5]
p0_amine = [0.05, 1.5, 2.5]
p0_hydroxyl = [0.05, 1.5, 0.6]
##Lower bounds for curve fitting##
lb_water = [0.02, 0.01, -1e-6]
lb_mt = [0.0, 30, -2.5]
lb_noe = [0.0, 0.5, -4.5]
lb_noe_neg_1_6 = [0.0, 0.5, -1.8]
lb_creatine = [0.0, 0.5, 1.6]
lb_amide = [0.0, 0.5, 3.2]
lb_amine = [0.0, 0.1, 2.2]
lb_hydroxyl = [0.0, 0.1, 0.4]
##Upper bounds for curve fitting##
ub_water = [1, 10, 1e-6]
ub_mt = [0.5, 60, 0]
ub_noe = [0.25, 5, -1.5]
ub_noe_neg_1_6 = [.25, 5, -1.2]
ub_creatine = [0.5, 5, 2.6]
ub_amide = [0.3, 5, 4.0]
ub_amine = [0.3, 5, 2.8]
ub_hydroxyl = [0.3, 5, 1.2]

##Combine for curve fitting##
#Step 1
p0_1 = p0_water + p0_mt
lb_1 = lb_water + lb_mt
ub_1 = ub_water + ub_mt 
#Step 2 (cardiac)
p0_2 = p0_noe + p0_creatine + p0_amide
lb_2 = lb_noe + lb_creatine + lb_amide
ub_2 = ub_noe + ub_creatine + ub_amide
#Single step (Cr phantom)
p0_ph = p0_water + p0_creatine
lb_ph = lb_water + lb_creatine
ub_ph = ub_water + ub_creatine


#Cutoffs and options for fitting
cutoffs = [-4, -1.4, 1.4, 4]
options = {'xtol': 1e-10, 'ftol': 1e-4, 'maxfev': 50}

def Lorentzian(x, Amp, Fwhm, Offset):
    Num = Amp * 0.25 * Fwhm ** 2
    Den = 0.25 * Fwhm ** 2 + (x - Offset) ** 2
    return Num/Den

def Step_1_Fit(x, *fit_parameters):
    Water_Fit = Lorentzian(x, fit_parameters[0], fit_parameters[1], fit_parameters[2])
    Mt_Fit = Lorentzian(x, fit_parameters[3], fit_parameters[4], fit_parameters[5])
    Fit = 1 - Water_Fit - Mt_Fit
    return Fit

# def Step_2_Fit(x, *fit_parameters):
#     Noe_Fit = Lorentzian(x, fit_parameters[0], fit_parameters[1], fit_parameters[2])
#     Creatine_Fit = Lorentzian(x, fit_parameters[3], fit_parameters[4], fit_parameters[5])
#     Amide_Fit = Lorentzian(x, fit_parameters[6], fit_parameters[7], fit_parameters[8])
#     Fit = Noe_Fit + Creatine_Fit + Amide_Fit
#     return Fit

def Water_Fit_Correction(x, *fit_parameters):
    Water_Fit = Lorentzian(x, fit_parameters[0], fit_parameters[1], fit_parameters[2])
    Fit = 1 - Water_Fit
    return Fit

def calc_spectra(imgs, session_state):
    spectra = {}
    organ = session_state.submitted_data['organ']
    if organ == 'Cardiac':
        labeled_segments = session_state.user_geometry['aha']
        mask = session_state.user_geometry['masks']['lv']
        for label, segment in labeled_segments.items():
            pixels = []
            segment_mask = np.zeros_like(mask)
            for coord in segment:
                segment_mask[coord[0], coord[1]] = 1
            for i in range(np.size(imgs, axis=2)):
                img = imgs[:,:,i]      
                img_seg = img*segment_mask
                img_seg = img_seg.flatten()[img_seg.flatten() != 0]
                pixels.append(img_seg)
            pixels = np.array(pixels)
            spectrum = np.mean(pixels, axis=1)
            spectra[label] = spectrum
    else:
        masks = session_state.user_geometry['masks']
        for label, roi in masks.items():
            pixels = []
            for i in range(np.size(imgs, axis=2)):
                img = imgs[:,:,i]
                img_roi = img*roi
                img_roi = img_roi.flatten()[img_roi.flatten() != 0]
                pixels.append(img_roi)
            pixels = np.array(pixels)
            spectrum = np.mean(pixels, axis=1)
            spectra[label] = spectrum
    session_state.processed_data['spectra'] = spectra
    return session_state

def calc_spectra_pixelwise(imgs, session_state):
    spectra = {}
    organ = session_state.submitted_data['organ']
    # Determine masks based on organ type
    if organ == 'Cardiac':
        masks = {'lv': session_state.user_geometry['masks']['lv']}
    else:
        masks = session_state.user_geometry['masks']
    # Process each mask
    for label, mask in masks.items():
        pixels = []  # Clear pixels for each label
        for i in range(imgs.shape[2]):  # Loop over image slices
            img = imgs[:, :, i]
            # Apply mask and filter non-zero pixels
            masked_pixels = img[mask]
            pixels.append(masked_pixels)
        # Stack and store pixel data
        pixels = np.array(pixels).T  # Transpose to swap axes
        spectra[label] = pixels.tolist()
    # Save spectra to session state
    session_state.processed_data["pixelwise"]["spectra"] = spectra

def two_step(spectra, offsets, contrasts):
    n_interp = 4000
    fits = {}
    for roi, spectrum_data in spectra.items():
        # Check if it's pixelwise data (list of lists) or ROI data (single list)
        if isinstance(spectrum_data[0], list):
            # Handle pixelwise spectra
            roi_fits = []
            for spectrum in spectrum_data:
                roi_fits.append(_process_spectrum(offsets, spectrum, n_interp, contrasts))
            fits[roi] = roi_fits
        else:
            # Handle ROI-averaged spectrum
            fits[roi] = _process_spectrum(offsets, spectrum_data, n_interp, contrasts)
    return fits

def _process_spectrum(offsets, spectrum, n_interp, custom_contrasts=None):
    if custom_contrasts is None:
        custom_contrasts = ['Amide', 'Creatine', 'NOE (-2.75 ppm)']

    contrast_params = {
        'NOE (-2.75 ppm)': (p0_noe, lb_noe, ub_noe),
        'Creatine': (p0_creatine, lb_creatine, ub_creatine),
        'Amide': (p0_amide, lb_amide, ub_amide),
        'Amine': (p0_amine, lb_amine, ub_amine),
        'Hydroxyl': (p0_hydroxyl, lb_hydroxyl, ub_hydroxyl),
        'NOE (-1.6 ppm)': (p0_noe_neg_1_6, lb_noe_neg_1_6, ub_noe_neg_1_6)

    }

    p0_2, lb_2, ub_2 = [], [], []
    for contrast in custom_contrasts:
        p0_2 += contrast_params[contrast][0]
        lb_2 += contrast_params[contrast][1]
        ub_2 += contrast_params[contrast][2]

    def Step_2_Fit(x, *params):
        fit_sum = np.zeros_like(x)
        index = 0
        for contrast in custom_contrasts:
            fit_sum += Lorentzian(x, params[index], params[index + 1], params[index + 2])
            index += 3
        return fit_sum

    try:
        if offsets[0] > 0:
            offsets = np.flip(offsets)
            spectrum = np.flip(spectrum)

        fit_1, _ = curve_fit(Step_1_Fit, offsets, spectrum, p0=p0_corr, bounds=(lb_corr, ub_corr), **options)
        correction = fit_1[2]
        offsets_corrected = offsets - correction

        if 'Hydroxyl' in custom_contrasts:
            cutoffs[2] = 0.4
        else:
            cutoffs[2] = 1.4

        condition = (offsets_corrected <= cutoffs[0]) | (offsets_corrected >= cutoffs[3]) | \
                    ((offsets_corrected >= cutoffs[1]) & (offsets_corrected <= cutoffs[2]))

        condition_rmse = ((offsets_corrected <= -1.4) & (offsets_corrected >= -4)) | \
                         ((offsets_corrected >= 1.4) & (offsets_corrected <= 4))

        offsets_cropped = offsets_corrected[condition]
        spectrum_cropped = spectrum[condition]

        if len(offsets_cropped) == 0:  # Handle empty offsets case
            raise RuntimeError("No valid offsets found after cropping")

        offsets_interp = np.linspace(offsets_corrected[0], offsets_corrected[-1], n_interp)

        fit_1, _ = curve_fit(Step_1_Fit, offsets_cropped, spectrum_cropped, p0=p0_1, bounds=(lb_1, ub_1), **options)
        water_fit = Lorentzian(offsets_interp, fit_1[0], fit_1[1], fit_1[2])
        mt_fit = Lorentzian(offsets_interp, fit_1[3], fit_1[4], fit_1[5])

        background = Lorentzian(offsets_corrected, fit_1[0], fit_1[1], fit_1[2]) + \
                     Lorentzian(offsets_corrected, fit_1[3], fit_1[4], fit_1[5])
        lorentzian_difference = 1 - (spectrum + background)

        step_1_fit_values = Step_1_Fit(offsets_corrected, *fit_1)
        step_1_rmse = np.sqrt(mean_squared_error(spectrum, step_1_fit_values))

        fit_2, _ = curve_fit(Step_2_Fit, offsets_corrected, lorentzian_difference, p0=p0_2, bounds=(lb_2, ub_2), **options)
        fit_curves = {}
        index = 0
        for contrast in custom_contrasts:
            fit_curves[contrast] = Lorentzian(offsets_interp, fit_2[index], fit_2[index + 1], fit_2[index + 2])
            index += 3

        step_2_fit_values = Step_2_Fit(offsets_corrected, *fit_2)
        step_2_rmse = np.sqrt(mean_squared_error(lorentzian_difference, step_2_fit_values))

        total_fit = step_1_fit_values - step_2_fit_values
        spectrum_region = spectrum[condition_rmse]
        total_fit_region = total_fit[condition_rmse]
        rmse = np.sqrt(mean_squared_error(spectrum_region, total_fit_region))

        offsets_interp = np.flip(offsets_interp)
        water_fit = np.flip(water_fit)
        mt_fit = np.flip(mt_fit)
        fit_curves_named = {f"{contrast}_Fit": np.flip(fit_curves[contrast]) for contrast in fit_curves}

        contrasts = {'Water': 100 * fit_1[0], 'MT': 100 * fit_1[3]}
        for i, contrast in enumerate(custom_contrasts):
            contrasts[contrast] = 100 * fit_2[i * 3]

        data_dict = {'Zspec': spectrum, 'Offsets': offsets, 'Offsets_Corrected': offsets_corrected,
                     'Offsets_Interp': offsets_interp, 'Water_Fit': water_fit, 'MT_Fit': mt_fit,
                     **fit_curves_named, 'Lorentzian_Difference': lorentzian_difference}

        fit_parameters = [fit_1, fit_2]

    except RuntimeError:
        # Assign zeros instead of crashing
        fit_parameters = [np.zeros(len(p0_1)), np.zeros(len(p0_2))]
        contrasts = {key: 0 for key in ['Water', 'MT'] + custom_contrasts}
        data_dict = {'Zspec': spectrum, 'Offsets': offsets, 'Offsets_Corrected': np.zeros_like(offsets),
                     'Offsets_Interp': np.zeros(n_interp), 'Water_Fit': np.zeros(n_interp), 'MT_Fit': np.zeros(n_interp),
                     'Lorentzian_Difference': np.zeros(n_interp), **{f"{contrast}_Fit": np.zeros(n_interp) for contrast in custom_contrasts}}

        spectrum_region = np.array([])
        total_fit_region = np.array([])
        rmse = np.inf

    return {'Fit_Params': fit_parameters, 'Data_Dict': data_dict,
            'Contrasts': contrasts, 'Residuals': spectrum_region - total_fit_region, 'RMSE': rmse}


# def _process_spectrum(offsets, spectrum, n_interp, custom_contrasts = None):
#     if custom_contrasts is None:
#         custom_contrasts = ['Amide', 'Creatine', 'NOE (-2.75 ppm)']
#     # Define fitting parameters dynamically
#     contrast_params = {
#         'NOE (-2.75 ppm)': (p0_noe, lb_noe, ub_noe),
#         'Creatine': (p0_creatine, lb_creatine, ub_creatine),
#         'Amide': (p0_amide, lb_amide, ub_amide),
#         'Amine': (p0_amine, lb_amine, ub_amine),
#         'Hydroxyl': (p0_hydroxyl, lb_hydroxyl, ub_hydroxyl)
#     }
#     p0_2, lb_2, ub_2 = [], [], []
#     for contrast in custom_contrasts:
#             p0_2 += contrast_params[contrast][0]
#             lb_2 += contrast_params[contrast][1]
#             ub_2 += contrast_params[contrast][2]
#     def Step_2_Fit(x, *params):
#         fit_sum = np.zeros_like(x)
#         index = 0
#         for contrast in custom_contrasts:
#             fit_sum += Lorentzian(x, params[index], params[index+1], params[index+2])
#             index += 3
#         return fit_sum
#     try:
#         if offsets[0] > 0:
#             offsets = np.flip(offsets)
#             spectrum = np.flip(spectrum)
#         # Step 1 fitting
#         fit_1, _ = curve_fit(Step_1_Fit, offsets, spectrum, p0=p0_corr, bounds=(lb_corr, ub_corr), **options)
#         correction = fit_1[2]
#         offsets_corrected = offsets - correction
#         # Exception for OH
#         if 'Hydroxyl' in custom_contrasts:
#             cutoffs[2] = 0.4
#         else:
#             cutoffs[2] = 1.4
#         # Crop offsets and spectrum
#         condition = (offsets_corrected <= cutoffs[0]) | (offsets_corrected >= cutoffs[3]) | \
#                     ((offsets_corrected >= cutoffs[1]) & (offsets_corrected <= cutoffs[2]))
#         condition_rmse = ((offsets_corrected <= -1.4) & (offsets_corrected >= -4)) | \
#                          ((offsets_corrected >= 1.4) & (offsets_corrected <= 4))
#         offsets_cropped = offsets_corrected[condition]
#         spectrum_cropped = spectrum[condition]
#         # Interpolated frequency axis
#         offsets_interp = np.linspace(offsets_corrected[0], offsets_corrected[-1], n_interp)
#         # Fit Step 1
#         fit_1, _ = curve_fit(Step_1_Fit, offsets_cropped, spectrum_cropped, p0=p0_1, bounds=(lb_1, ub_1), **options)
#         water_fit = Lorentzian(offsets_interp, fit_1[0], fit_1[1], fit_1[2])
#         mt_fit = Lorentzian(offsets_interp, fit_1[3], fit_1[4], fit_1[5])
#         # Background and Lorentzian difference
#         background = Lorentzian(offsets_corrected, fit_1[0], fit_1[1], fit_1[2]) + \
#                      Lorentzian(offsets_corrected, fit_1[3], fit_1[4], fit_1[5])
#         lorentzian_difference = 1 - (spectrum + background)
#         # RMSE for Step 1
#         step_1_fit_values = Step_1_Fit(offsets_corrected, *fit_1)
#         step_1_rmse = np.sqrt(mean_squared_error(spectrum, step_1_fit_values))
#         # Step 2 fitting
#         fit_2, _ = curve_fit(Step_2_Fit, offsets_corrected, lorentzian_difference, p0=p0_2, bounds=(lb_2, ub_2), **options)
#         fit_curves = {}
#         index = 0
#         for contrast in custom_contrasts:
#             fit_curves[contrast] = Lorentzian(offsets_interp, fit_2[index], fit_2[index+1], fit_2[index+2])
#             index += 3
#         # RMSE for Step 2
#         step_2_fit_values = Step_2_Fit(offsets_corrected, *fit_2)
#         step_2_rmse = np.sqrt(mean_squared_error(lorentzian_difference, step_2_fit_values))
#         # Total fit and final RMSE
#         total_fit = step_1_fit_values - step_2_fit_values
#         spectrum_region = spectrum[condition_rmse]
#         total_fit_region = total_fit[condition_rmse]
#         rmse = np.sqrt(mean_squared_error(spectrum_region, total_fit_region))
#         # Flip for NMR convention
#         offsets_interp = np.flip(offsets_interp)
#         water_fit = np.flip(water_fit)
#         mt_fit = np.flip(mt_fit)
#         fit_curves_named = {}
#         for contrast in fit_curves:
#             fit_curves_named[f"{contrast}_Fit"] = np.flip(fit_curves[contrast])

#         contrasts = {'Water': 100 * fit_1[0], 'MT': 100 * fit_1[3]}
#         for i, contrast in enumerate(custom_contrasts):
#             contrasts[contrast] = 100 * fit_2[i * 3]
#         # Prepare result
#         data_dict = {'Zspec': spectrum, 'Offsets': offsets, 'Offsets_Corrected': offsets_corrected,
#                            'Offsets_Interp': offsets_interp, 'Water_Fit': water_fit, 'MT_Fit': mt_fit,
#                            **fit_curves_named,'Lorentzian_Difference': lorentzian_difference}
#         fit_parameters = [fit_1, fit_2]
#     except RuntimeError:
#         fit_parameters = [np.zeros(len(p0_1)), np.zeros(len(p0_2))]
#         contrasts = {key: 0 for key in ['Water', 'MT'] + custom_contrasts}
#         data_dict = {'Zspec': spectrum, 'Offsets': offsets, 'Offsets_Corrected': offsets_corrected,
#                      'Offsets_Interp': offsets_interp, 'Water_Fit': np.zeros(n_interp), 'MT_Fit': np.zeros(n_interp),
#                      'Lorentzian_Difference': np.zeros(n_interp), **{key: np.zeros(n_interp) for key in custom_contrasts}}
#         spectrum_region = np.array([])
#         total_fit_region = np.array([])
#         rmse = np.inf
    
#     return {'Fit_Params': fit_parameters, 'Data_Dict': data_dict,
#             'Contrasts': contrasts, 'Residuals': spectrum_region - total_fit_region, 'RMSE': rmse}

def per_pixel(session_state):
    fits = {}
    contrasts = session_state.custom_contrasts
    spectra = session_state.processed_data['pixelwise']['spectra']
    offsets = session_state.recon['cest']['offsets']

    # Initialize the progress bar
    total = sum(len(pixels) for pixels in spectra.values())
    progress_bar = st.progress(0)
    progress_counter = 0

    for label, pixels in spectra.items():
        fits[label] = []
        for spectrum in pixels:
            result = two_step({label: [spectrum]}, offsets, contrasts)
            fits[label].append(result[label][0])  # Assuming result[label] is a list with a single dictionary
            progress_counter += 1
            # Update the progress bar
            progress_bar.progress(progress_counter / total, text="Performing pixelwise fitting...")
    
    # Mark the progress bar as complete
    progress_bar.progress(1.0, text="Fitting complete.")
    # Remove progress bar after full
    progress_bar.empty()
    return fits

def wassr(offsets, spectra):
    pixelwise = []
    n_interp = 1000  # Number of points for interpolation
    for spectrum in spectra:
        if offsets[0] > 0:
            offsets = np.flip(offsets)
            spectrum = np.flip(spectrum)
        # Interpolate offsets and spectrum using cubic spline
        cubic_spline = CubicSpline(offsets, spectrum)
        offsets_interp = np.linspace(offsets[0], offsets[-1], n_interp)
        spectrum_interp = cubic_spline(offsets_interp)
        # Fit for corrections
        Fit_1, _ = curve_fit(Step_1_Fit, offsets_interp, spectrum_interp, p0=p0_corr, bounds=(lb_corr, ub_corr))
        # Calculate water and MT fits from parameters
        Water_Fit = Lorentzian(offsets_interp, Fit_1[0], Fit_1[1], Fit_1[2])
        Mt_Fit = Lorentzian(offsets_interp, Fit_1[3], Fit_1[4], Fit_1[5])
        # B0 correction: find offset corresponding to the peak
        b0_shift = offsets_interp[np.argmax(Water_Fit + Mt_Fit)]
        pixelwise.append(b0_shift)
    return pixelwise