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

