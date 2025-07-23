#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 12:56:50 2024

@author: jonah
"""
import os
import base64
import io
import math
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from streamlit_drawable_canvas import st_canvas
from PIL import Image, ImageDraw
from scipy.ndimage import zoom
from skimage.filters import threshold_otsu, median
from skimage import morphology
from skimage.transform import  resize

# --- Calculation functions --- #
def distance(co1, co2):
    """
    Calculate squared Euclidean distance.
    """
    return abs(co1[0] - co2[0])**2 + abs(co1[1] - co2[1])**2

def centroid(array):
    """
    Calculate centroid.
    """
    x_c = 0
    y_c = 0
    area = array.sum()
    it = np.nditer(array, flags=['multi_index'])
    for i in it:
        x_c = i * it.multi_index[1] + x_c
        y_c = i * it.multi_index[0] + y_c
    return (int(x_c / area), int(y_c / area))

def convert_rois_to_masks(image, rois):
    """
    Converts ROI path data from the canvas into boolean masks.
    """
    masks = {}
    image_shape = np.shape(image)
    for name, path_data in rois.items():
        # Extract coordinates from path data
        polygon = []
        for command in path_data:
            if command[0] in ["M", "L"]:  # "Move" or "Line" commands
                x, y = command[1], command[2]
                polygon.append((x, y))  # Add coordinates as a tuple
            elif command[0] == "z":  # Close path command, ignored for drawing
                pass
        # Create a blank mask
        mask = Image.new("L", (image_shape[1], image_shape[0]), 0)
        # Handle insertion_points ROI differently
        if name == "insertion_points" and len(polygon) == 2:
            # Only mark the first and last points
            for point in [polygon[0], polygon[-1]]:
                mask.putpixel(point, 1)
        elif polygon:  # For other ROIs, draw the polygon as usual
            ImageDraw.Draw(mask).polygon(polygon, outline=1, fill=1)
        # Convert the mask to a numpy array
        masks[name] = np.array(mask).astype(bool)
    return masks

def calc_lv_mask(masks):
    """
    Calculates left ventricle from epi/endocardial masks.
    """
    if 'epicardium' not in masks or 'endocardium' not in masks:
        return np.zeros_like(next(iter(masks.values())), dtype=bool)
    return np.logical_and(masks['epicardium'], np.logical_not(masks['endocardium']))

def aha_segmentation(mask, ip_mask):
    """
    Performs AHA segmentation on the myocardium using LV and RV insertion point masks.
    """
    mask_coords = np.argwhere(mask)
    ip_coords = np.argwhere(ip_mask)
    ip_coords = np.array([ip_coords[0], ip_coords[-1]])
    # Get points in myocardium with closest proximity to defined insertion points 
    insertion_points = []
    for coord in ip_coords:
        closest = mask_coords[0]
        for c in mask_coords:
            if distance(c, coord) < distance(closest, coord):
                closest = c
        insertion_points.append(closest)
    arv = insertion_points[0]
    irv = insertion_points[1]
    cx, cy = centroid(mask)
    [y, x] = np.nonzero(mask)
    inds = np.nonzero(mask)
    inds = list(zip(inds[0], inds[1]))
    # Offset all points by centroid
    x = x - cx
    y = y - cy
    arvx = arv[1] - cx
    arvy = arv[0] - cy
    irvx = irv[1] - cx
    irvy = irv[0] - cy
    # Find angular segment cutoffs
    pi = math.pi
    angle = lambda a, b: (math.atan2(a, b)) % (2 * pi)
    arv_ang = angle(arvy, arvx)
    irv_ang = angle(irvy, irvx)
    ang = [angle(yc, xc) for yc, xc in zip(y, x)]
    sept_cutoffs = np.linspace(0, arv_ang - irv_ang, num=3)  # two septal segments
    wall_cutoffs = np.linspace(arv_ang - irv_ang, 2 * pi, num=5)  # four wall segments
    cutoffs = []
    cutoffs.extend(sept_cutoffs)
    cutoffs.extend(wall_cutoffs[1:])
    ang = [(a - irv_ang) % (2 * pi) for a in ang]
    # Create arrays of each pixel/index in each segment
    segment_image = lambda a, b: [j for (i, j) in enumerate(inds) if ang[i] >= a and ang[i] < b]
    segmented_indices = [segment_image(a, b) for a, b in zip(cutoffs[:6], cutoffs[1:])]
    # List of labeled segments
    labeled_segments = {}
    labeled_segments['Inferoseptal'] = segmented_indices[0]
    labeled_segments['Anteroseptal'] = segmented_indices[1]
    labeled_segments['Anterior'] = segmented_indices[2]
    labeled_segments['Anterolateral'] = segmented_indices[3]
    labeled_segments['Inferolateral'] = segmented_indices[4]
    labeled_segments['Inferior'] = segmented_indices[5]
    return labeled_segments

# --- Helper functions for the UI --- #    
def get_base64_image(image_path):
    """
    Convert image to base64 encoded string.
    """
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")

@st.cache_data
def _prepare_canvas_background(bg_image_data, size_ref_data=None):
    """
    Prepares the m0 image to be used as a background for the canvas.
    Resizes the background image to match reference data if necessary.
    """
    display_image = bg_image_data.squeeze()
    # If a size reference is provided and the shapes don't match, resize the background image.
    if size_ref_data is not None:
        size_ref_img = size_ref_data.squeeze()
        if display_image.shape != size_ref_img.shape:
            zoom_factors = np.array(size_ref_img.shape) / np.array(display_image.shape)
            display_image = zoom(display_image, zoom_factors, order=2)  # Bilinear interpolation
    # All subsequent calculations use the (potentially resized) display_image.
    img_height, img_width = display_image.shape
    canvas_width = 600
    canvas_height = int(canvas_width * img_height / img_width)
    fig, ax = plt.subplots(figsize=(canvas_width/100, canvas_height/100))
    ax.imshow(display_image, cmap='gray')
    ax.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    img_byte_arr = io.BytesIO()
    fig.savefig(img_byte_arr, format='PNG', dpi=100)
    img_byte_arr.seek(0)
    plt.close(fig)
    return Image.open(img_byte_arr), (img_width, img_height), (canvas_width, canvas_height)


# --- Interactive UI functions --- #
def draw_rois(image, size_ref_image=None):
    """
    UI component for drawing general-purpose ROIs.
    Returns the final ROI data upon user submission.
    """
    background_image, (img_w, img_h), (can_w, can_h) = _prepare_canvas_background(image, size_ref_image)
    
    st.markdown("## ROI Instructions")
    st.markdown("**Please read carefully!!**")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    icons_dir = os.path.join(script_dir, "../custom/icons")
    download = get_base64_image(os.path.join(icons_dir, "download.png"))
    undo = get_base64_image(os.path.join(icons_dir, "undo.png"))
    trash = get_base64_image(os.path.join(icons_dir, "bin.png"))

    col1, col2 = st.columns((2, 1))
    with col1:
        st.markdown(f"""
        1. **Draw as many ROIs as you like!** <ul>
                <li> Multiple ROIs (e.g., for a phantom) and single ROIs (e.g., for liver) are both ok!</li>
                <li> Close each ROI when finished.</li>
            </ul>
        2. **Label each ROI** using the text entry field.
        3. When finished, click <img src="data:image/jpeg;base64,{download}" style="display:inline; width:20px;" />.
        4. Finally, click **Submit ROI(s)**.
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <br>
        <div style="border: 2px solid #ddd; padding: 10px; border-radius: 5px; background-color: #f9f9f9;">
            <strong>Tips</strong><br>
            <ul>
                <li>Right-click to close a polygon.</li>
                <li>Click <img src="data:image/jpeg;base64,{undo}" style="display:inline; width:20px;" /> to undo the last action.</li>
                <li>Click <img src="data:image/jpeg;base64,{trash}" style="display:inline; width:20px;" /> to delete all polygons.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    canvas_result = st_canvas(
        fill_color="rgba(0, 0, 0, 0)", stroke_color="yellow", stroke_width=2,
        height=can_h, width=can_w, drawing_mode='polygon',
        background_image=background_image, update_streamlit=False,
        display_toolbar=True, key="canvas_other"
    )

    # Access the coordinates of the drawn polygon from the canvas_result
    if canvas_result.json_data is not None:
            polygons = canvas_result.json_data.get("objects", [])
            rois = {}
            for idx, polygon in enumerate(polygons):
                name = st.text_input(f"Name ROI {idx+1}", f"ROI {idx+1}")
                coords = polygon['path']
                # Rescale coordinates back to the original image dimensions
                scale_x = img_width / canvas_width  # Scaling factor for width
                scale_y = img_height / canvas_height  # Scaling factor for height
                
                original_coords = []
                for i, cmd in enumerate(coords):
                    if cmd[0] in ["M", "L"]:  # Move/Line commands
                        x_scaled = int(cmd[1] * scale_x)
                        y_scaled = int(cmd[2] * scale_y)
                        original_coords.append([cmd[0], x_scaled, y_scaled])
                    elif cmd[0] == "z":  # Close path
                        original_coords.append([cmd[0]])
                
                rois[name] = original_coords
            session_state.user_geometry["rois"] = rois
                
                #st.write("Original Coordinates:", original_coords) # For debugging
                #st.write(names) # For debugging
            if st.button("Submit ROI(s)"):
                session_state.rois_done = True
                st.rerun()
#take out mask_key resitricted to ROI 1
def auto_segment_hydrogel(session_state): # No mask_key parameter
    """
    Refines all masks in session_state.user_geometry['masks'] by applying threshold-based
    segmentation and smoothing, based on the provided image processing logic.
    Overwrites the initial masks with the refined versions in session_state.

    This version processes all masks without specific exclusions or advanced error handling.

    Parameters:
    - session_state: Streamlit session_state object.
    """
    if session_state.reference is None:
        st.error("Reference image is required for auto-segmentation.")
        return

    reference_image_raw = np.squeeze(session_state.reference)
    
    # Work on a copy of the masks to avoid issues if the dictionary changes during iteration
    all_masks_in_session_state = session_state.user_geometry['masks'].copy() 

    segmented_count = 0
    # Iterate through all masks in the current session state
    for mask_label, current_mask_data in all_masks_in_session_state.items():
        if current_mask_data is None:
            print(f"[WARNING] Mask '{mask_label}' is None. Skipping auto-segmentation for this mask.")
            continue

        # Core logic from your provided auto_segment_hydrogel snippet:
        image = np.copy(reference_image_raw) # Work on a copy of the reference image
        mask = np.squeeze(current_mask_data) # Squeeze the current mask

        # Ensure shapes match by resizing image to mask's shape
        if image.shape != mask.shape:
            print(f"[INFO] Resizing reference image from {image.shape} to match mask '{mask_label}' ({mask.shape}) for auto-segmentation.")
            image = resize(image, mask.shape, preserve_range=True, anti_aliasing=True)
        
        # Assert is for debugging; might be removed in production code if you prefer soft error handling
        assert image.shape == mask.shape, f"Shape mismatch after resize: {image.shape} vs {mask.shape} for mask '{mask_label}'"

        # Apply the mask to the image (this isolates the region of interest)
        masked_image = image * mask
        
        # Perform Otsu thresholding directly on the masked image
        # Note: This does not include checks for `masked_image` being all zeros or constant,
        # which could lead to errors if `threshold_otsu` can't find a threshold.
        thresh = threshold_otsu(masked_image)
        binary = masked_image > thresh
        smoothed_binary = median(binary, footprint=morphology.disk(4))
        
        # Overwrite the original mask in session state with the refined result
        refined_mask = smoothed_binary.astype(np.uint8)
        session_state.user_geometry['masks'][mask_label] = refined_mask

        print(f"[SUCCESS] Auto-segmentation complete. Mask '{mask_label}' updated. Final mask shape: {refined_mask.shape}, Non-zero pixels: {np.count_nonzero(refined_mask)}")
        segmented_count += 1
    
    if segmented_count > 0:
        st.success(f"Auto-segmentation applied to {segmented_count} ROI(s).")
    else:
        st.info("No masks found or processed for auto-segmentation.")

def create_multi_zone_masks(session_state, base_mask_key='ROI 1'):
    """
    Creates multiple concentric ring masks (Zone 1 and Zone 2) around a specified base mask.
    Zone 0 is implicitly the base_mask_key itself (e.g., 'ROI 1').
    The thickness of each zone is determined by 'ring_dilation_pixels' from session_state.
    Parameters:
    - session_state: Streamlit session_state object.
    - base_mask_key (str): The key of the base mask in session_state.user_geometry['masks']
                           (e.g., 'ROI 1', which should be the auto-segmented hydrogel).
    """
    # Ensure the reference image is available for shape validation
    if session_state.reference is None:
        st.error("Reference image is required to create multi-zone masks.")
        return
    # Retrieve the base mask (expected to be the auto-segmented ROI 1)
    base_mask = session_state.user_geometry['masks'].get(base_mask_key)
    if base_mask is None:
        st.warning(f"Base mask '{base_mask_key}' not found. Cannot create multi-zone masks.")
        return
    # Ensure base mask is a 2D boolean array
    base_mask_2d = np.squeeze(base_mask).astype(bool)
    mask_original_shape = base_mask_2d.shape # Store original shape for final resize
    # Check if the base mask is empty
    if not np.any(base_mask_2d):
        st.warning(f"Base mask '{base_mask_key}' is empty. Cannot create meaningful multi-zone masks.")
        # Store empty masks for the zones to avoid KeyErrors downstream if they are expected
        session_state.user_geometry['masks']['ring_zone_1'] = np.zeros_like(base_mask_2d, dtype=np.uint8)
        session_state.user_geometry['masks']['ring_zone_2'] = np.zeros_like(base_mask_2d, dtype=np.uint8)
        return

    # Ensure mask shape matches reference image slice shape for consistency during dilation
    # Assuming reference image is 2D (rows, cols)
    ref_image_shape = session_state.reference.shape[:2] if session_state.reference.ndim >=2 else session_state.reference.shape

    if base_mask_2d.shape != ref_image_shape:
        print(f"[WARNING] Base mask '{base_mask_key}' shape {base_mask_2d.shape} does not match reference image shape {ref_image_shape}. Resizing base mask for zone creation.")
        base_mask_2d = resize(base_mask_2d, ref_image_shape, order=0, preserve_range=True, anti_aliasing=False)
        base_mask_2d = base_mask_2d.astype(bool)

    # Get the dilation pixel value from session state
    # Provide a default (e.g., 5) in case the key isn't found for some reason
    dilation_pixels = session_state.submitted_data.get('ring_dilation_pixels', 5) 
    # Define the structural element for dilation
    selem = morphology.disk(dilation_pixels)
    # --- Generate Zone 1 (First Ring) ---
    # Dilate the base mask (which might have been resized to ref_image_shape)
    dilated_mask_1 = morphology.dilation(base_mask_2d, selem)
    # Zone 1 is the region in dilated_mask_1 but not in base_mask_2d
    ring_zone_1_mask = dilated_mask_1 & ~base_mask_2d
    # Resize back to the original mask shape before storing
    ring_zone_1_mask = resize(ring_zone_1_mask, mask_original_shape, order=0, preserve_range=True, anti_aliasing=False)
    session_state.user_geometry['masks']['ring_zone_1'] = ring_zone_1_mask.astype(np.uint8)
    print(f"[INFO] Ring Zone 1 mask created. Shape: {ring_zone_1_mask.shape}, Non-zero pixels: {np.count_nonzero(ring_zone_1_mask)}")
    # --- Generate Zone 2 (Second Ring) ---
    # The base for Zone 2 starts from the outer edge of Zone 1, which is `dilated_mask_1`
    dilated_mask_2 = morphology.dilation(dilated_mask_1, selem)
    # Zone 2 is the region in dilated_mask_2 but not in dilated_mask_1
    ring_zone_2_mask = dilated_mask_2 & ~dilated_mask_1
    # Resize back to the original mask shape before storing
    ring_zone_2_mask = resize(ring_zone_2_mask, mask_original_shape, order=0, preserve_range=True, anti_aliasing=False)
    session_state.user_geometry['masks']['ring_zone_2'] = ring_zone_2_mask.astype(np.uint8)
    print(f"[INFO] Ring Zone 2 mask created. Shape: {ring_zone_2_mask.shape}, Non-zero pixels: {np.count_nonzero(ring_zone_2_mask)}")
    st.success(f"Multi-layered Spatial Zone analysis masks (Zone 1 & Zone 2) created with thickness {dilation_pixels} pixels.")



def cardiac_roi(session_state, data, cest):
    # Load images
    # Get the directory of the current script
    if canvas_result.json_data and canvas_result.json_data.get("objects"):
        polygons = canvas_result.json_data["objects"]
        rois = {}
        for idx, polygon in enumerate(polygons):
            name = st.text_input(f"Name ROI {idx+1}", f"ROI {idx+1}", key=f"roi_name_{idx}")
            coords = polygon['path']
            scale_x = img_w / can_w
            scale_y = img_h / can_h
            original_coords = [[cmd[0]] if cmd[0] == "z" else [cmd[0], int(cmd[1] * scale_x), int(cmd[2] * scale_y)] for cmd in coords]
            rois[name] = original_coords

        if st.button("Submit ROI(s)", key="submit_other_rois"):
            return rois
    return None

def cardiac_roi(image, size_ref_image=None):
    """
    UI component for the specific cardiac ROI workflow.
    Returns the final ROI data upon user submission.
    """
    background_image, (img_w, img_h), (can_w, can_h) = _prepare_canvas_background(image, size_ref_image)

    st.markdown("## Cardiac ROI Instructions")
    st.markdown("**Please read carefully!! If order of operations is not respected you will have to redraw ROIs!**")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    icons_dir = os.path.join(script_dir, "../custom/icons")
    download = get_base64_image(os.path.join(icons_dir, "download.png"))
    undo = get_base64_image(os.path.join(icons_dir, "undo.png"))
    trash = get_base64_image(os.path.join(icons_dir, "bin.png"))
    
    col1, col2 = st.columns((1, 1))
    with col1:
        st.markdown(f"""
        1. Draw a line connecting RV insertion points **anterior** to **inferior**.
        2. Draw the **epicardial** boundary for the LV myocardium.
        3. Draw the **endocardial** boundary for the LV myocardium.
        4. When finished, click <img src="data:image/jpeg;base64,{download}" style="display:inline; width:20px;" />.
        5. Finally, click *Submit ROI(s)*.
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div style="border: 2px solid #ddd; padding: 10px; border-radius: 5px; background-color: #f9f9f9;">
            <strong>Tips</strong><br>
            <ul>
                <li>Right-click to close a polygon.</li>
                <li>Click <img src="data:image/jpeg;base64,{undo}" style="display:inline; width:20px;" /> to undo the last action.</li>
                <li>Click <img src="data:image/jpeg;base64,{trash}" style="display:inline; width:20px;" /> to delete all polygons.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    canvas_result = st_canvas(
        fill_color="rgba(0, 0, 0, 0)", stroke_color="yellow", stroke_width=2,
        height=can_h, width=can_w, drawing_mode='polygon',
        background_image=background_image, update_streamlit=False,
        display_toolbar=True, key="canvas_cardiac"
    )

    if canvas_result.json_data and canvas_result.json_data.get("objects"):
        polygons = canvas_result.json_data["objects"]
        
        if len(polygons) < 3:
            st.warning(f"Waiting for {3 - len(polygons)} more ROIs!")
        elif len(polygons) > 3:
            st.error("Too many ROIs! Cardiac segmentation requires exactly 3.")
        else:
            first_poly_points = [cmd for cmd in polygons[0]['path'] if cmd[0] in ["M", "L"]]
            if len(first_poly_points) != 2:
                st.error("The first ROI must be a line connecting two points (RV insertion points).")
            else:
                if st.button("Submit ROI(s)", key="submit_cardiac_rois"):
                    roi_coords = []
                    for polygon in polygons:
                        coords = polygon['path']
                        scale_x = img_w / can_w
                        scale_y = img_h / can_h
                        original_coords = [[cmd[0]] if cmd[0] == "z" else [cmd[0], int(cmd[1] * scale_x), int(cmd[2] * scale_y)] for cmd in coords]
                        roi_coords.append(original_coords)
                    
                    return {'insertion_points': roi_coords[0], 'epicardium': roi_coords[1], 'endocardium': roi_coords[2]}
    return None

