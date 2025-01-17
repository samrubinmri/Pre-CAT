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
import pickle
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from streamlit_drawable_canvas import st_canvas
from PIL import Image

def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")
    
# def calculate_cardiac_masks(session_state):
    # polygons = 
    
    
def draw_rois(session_state, data):
    # Load images
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    icons_dir = os.path.join(script_dir, "../custom/icons")  # Relative path to icons folder
    download_path = os.path.join(icons_dir, "download.png")
    undo_path = os.path.join(icons_dir, "undo.png")
    trash_path = os.path.join(icons_dir, "bin.png")
    # Load images
    download = get_base64_image(download_path)
    undo = get_base64_image(undo_path)
    trash = get_base64_image(trash_path)

    # Get image data for ROI
    m0 = data['m0']  # Replace with your actual image data (numpy array)

    # Get image dimensions (height, width)
    img_height, img_width = np.shape(m0)

    # Desired canvas size
    canvas_width = 800  # Set desired width for the canvas
    canvas_height = int(canvas_width * img_height / img_width)  # Maintain aspect ratio

    # Convert image to PIL object
    fig = plt.figure(figsize=(canvas_width / 100, canvas_height / 100))  # Inches for fig size
    ax = fig.add_axes([0, 0, 1, 1])  # Position the image to fill the entire figure
    
    ax.imshow(m0, cmap='gray')
    ax.axis('off')  # Turn off axes to remove any borders or ticks

    # Remove all extra space/margins
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # No padding around the image
    
    # Make sure the image fills the entire space
    fig.canvas.draw()  # Ensure the figure is rendered
    image = np.array(fig.canvas.renderer.buffer_rgba())  # Convert figure to numpy array
    pillow_image = Image.fromarray(image)  # Convert to Pillow image

    # Save the Pillow image to an in-memory file
    img_byte_arr = io.BytesIO()
    pillow_image.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)  # Reset pointer to the beginning of the byte array

    # Instructions and help
    st.markdown("""
    ## ROI Instructions
    **Please read carefully!!**
    """)
    col1, col2 = st.columns((2, 1))
    with col1:
        st.markdown(f"""
        1. **Draw as many ROIs as you like!**  
            <ul>
                <li> Multiple ROIs (e.g., for a phantom) and single ROIs (e.g., for liver) are both ok!</li>
                <li> ROIs within another ROI are also ok, but no logical operations will be performed.</li>
                <li> Close each ROI when finished.</li>
            </ul>
        2. **Label each ROI** using the text entry field (e.g., "50mM", "25mM").
        3. **When finished and satisfied, click** <img src="data:image/jpeg;base64,{download}" style="display:inline; width:20px;" />.
        4. Finally, click **Submit ROI(s)**.
        """, unsafe_allow_html=True)

    # Tips column with a box
    with col2:
        st.markdown("""
        <br>
        <div style="border: 2px solid #ddd; padding: 10px; border-radius: 5px; background-color: #f9f9f9;">
            <strong>Tips</strong><br>
            <ul>
                <li>Right-click to close a polygon.</li>
                <li>Click <img src="data:image/jpeg;base64,{undo}" style="display:inline; width:20px;" /> to undo the last action.</li>
                <li>Click <img src="data:image/jpeg;base64,{trash}" style="display:inline; width:20px;" /> to delete all polygons.</li>
            </ul>
        </div>
        """.format(undo=undo, trash=trash), unsafe_allow_html=True)

    # Canvas configuration to ensure the image fills the entire space
    canvas_result = st_canvas(
        fill_color="rgba(0, 0, 0, 0)", 
        stroke_color="yellow", 
        stroke_width=2, 
        height=canvas_height,  # Use calculated height to maintain aspect ratio
        width=canvas_width,     # Use fixed width
        drawing_mode='polygon', 
        background_image=Image.open(img_byte_arr),
        update_streamlit = False,
        display_toolbar=True,
        key="canvas"
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
                st.write(rois)
            session_state.user_geometry["rois"] = rois
                
                #st.write("Original Coordinates:", original_coords) # For debugging
                #st.write(names) # For debugging
            if st.button("Submit ROI(s)"):
                session_state.rois_done = True
                st.rerun()
    
def cardiac_roi(session_state, data):
    # Load images
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    icons_dir = os.path.join(script_dir, "../custom/icons")  # Relative path to icons folder
    download_path = os.path.join(icons_dir, "download.png")
    undo_path = os.path.join(icons_dir, "undo.png")
    trash_path = os.path.join(icons_dir, "bin.png")
    # Load images
    download = get_base64_image(download_path)
    undo = get_base64_image(undo_path)
    trash = get_base64_image(trash_path)

    # Get image data for ROI
    m0 = data['m0']  # Replace with your actual image data (numpy array)

    # Get image dimensions (height, width)
    img_height, img_width = np.shape(m0)

    # Desired canvas size
    canvas_width = 800  # Set desired width for the canvas
    canvas_height = int(canvas_width * img_height / img_width)  # Maintain aspect ratio

    # Convert image to PIL object
    fig = plt.figure(figsize=(canvas_width / 100, canvas_height / 100))  # Inches for fig size
    ax = fig.add_axes([0, 0, 1, 1])  # Position the image to fill the entire figure
    
    ax.imshow(m0, cmap='gray')
    ax.axis('off')  # Turn off axes to remove any borders or ticks

    # Remove all extra space/margins
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # No padding around the image
    
    # Make sure the image fills the entire space
    fig.canvas.draw()  # Ensure the figure is rendered
    image = np.array(fig.canvas.renderer.buffer_rgba())  # Convert figure to numpy array
    pillow_image = Image.fromarray(image)  # Convert to Pillow image

    # Save the Pillow image to an in-memory file
    img_byte_arr = io.BytesIO()
    pillow_image.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)  # Reset pointer to the beginning of the byte array

    # Instructions and help
    st.markdown("""
    ## Cardiac ROI Instructions
    **Please read carefully!! If order of operations is not respected you will have to redraw ROIs!**
    """)
    col1, col2 = st.columns((1, 1))
    with col1:
        st.markdown(f"""
        1. Draw a line connecting RV insertion points **anterior** to **inferior**.
        2. Draw the **epicardial** boundary for the LV myocardium.
        3. Draw the **endocardial** boundary for the LV myocardium.
        4. When finished and satisfied, click <img src="data:image/jpeg;base64,{download}" style="display:inline; width:20px;" />.
        5. Finally, click *Submit ROI(s)*.
        """, unsafe_allow_html=True)

    # Tips column with a box
    with col2:
        st.markdown("""
        <div style="border: 2px solid #ddd; padding: 10px; border-radius: 5px; background-color: #f9f9f9;">
            <strong>Tips</strong><br>
            <ul>
                <li>Right-click to close a polygon.</li>
                <li>Click <img src="data:image/jpeg;base64,{undo}" style="display:inline; width:20px;" /> to undo the last action.</li>
                <li>Click <img src="data:image/jpeg;base64,{trash}" style="display:inline; width:20px;" /> to delete all polygons.</li>
            </ul>
        </div>
        """.format(undo=undo, trash=trash), unsafe_allow_html=True)

    # Canvas configuration to ensure the image fills the entire space
    canvas_result = st_canvas(
        fill_color="rgba(0, 0, 0, 0)", 
        stroke_color="yellow", 
        stroke_width=2, 
        height=canvas_height,  # Use calculated height to maintain aspect ratio
        width=canvas_width,     # Use fixed width
        drawing_mode='polygon', 
        background_image=Image.open(img_byte_arr),
        update_streamlit = False,
        display_toolbar=True,
        key="canvas"
    )

    # Initialize session state for polygon count if not already set
    if "polygon_count" not in st.session_state:
        st.session_state.polygon_count = 0
    
    # Access the coordinates of the drawn polygon from the canvas_result
    if canvas_result.json_data is not None:
        polygons = canvas_result.json_data.get("objects", [])
        roi_coords = []
        valid_first_polygon = True
        
        if len(polygons) > 0:
            first_polygon = polygons[0]
            coords = first_polygon['path']
            # Ensure the first polygon is a line (length of coordinates should be 2)
            if len([cmd for cmd in coords if cmd[0] in ["M", "L"]]) != 2:
                valid_first_polygon = False
                st.error("The first ROI must be a line connecting two points (i.e., RV insertion points).")
        
        # Update the polygon count in session_state
        st.session_state.polygon_count = len(polygons)
    
        for polygon in polygons:
            # Get the coordinates from the path
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
            roi_coords.append(original_coords)
                                
                
        if st.session_state.polygon_count < 3:
            st.warning(f"Waiting for {3 - st.session_state.polygon_count} more ROIs!")
        elif st.session_state.polygon_count > 3:
            st.warning(f"Too many ROIs! You have {st.session_state.polygon_count} ROIs and cardiac segmentation requires 3.")
        elif st.session_state.polygon_count == 3 and valid_first_polygon:
            if st.button("Submit ROI(s)", key="cardiac_rois"):
                session_state.rois_done = True
                rois = {'insertion_points': roi_coords[0], 'epicardium': roi_coords[1], 'endocardium': roi_coords[2]}
                st.session_state.user_geometry["rois"] = rois
                st.rerun()