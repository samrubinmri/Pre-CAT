#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 10:36:34 2025

@author: jonah
"""

import streamlit as st
import pickle
import os

# Define CSS and HTML for hover effect
def add_hoverable_title_with_image(title_text, image_url):
    # CSS for the hover effect
    hover_css = f"""
        <style>
        .hoverable-title {{
            position: relative;
            display: inline-block;
            cursor: pointer;
            color: black; /* Ensure text color matches the main title */
            font-weight: bold;
            font-size: 3rem; /* Match main title font size */
        }}
        .hoverable-title .image-tooltip {{
            visibility: hidden;
            opacity: 0;
            position: absolute;
            top: 120%; /* Adjust tooltip position */
            left: 50%;
            transform: translateX(-50%);
            transition: opacity 0.3s, visibility 0.3s;
            z-index: 999;
            background: white;
            padding: 5px;
            border: 1px solid #ddd;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
            border-radius: 8px;
        }}
        .hoverable-title:hover .image-tooltip {{
            visibility: visible;
            opacity: 1;
        }}
        .image-tooltip img {{
            max-width: 200px; /* Adjust the image size */
            border-radius: 8px;
        }}
        </style>
    """

    # HTML structure for the title with hoverable image
    hover_html = f"""
        <div class="hoverable-title">
            {title_text}
            <div class="image-tooltip">
                <img src="{image_url}" alt="Tooltip Image">
            </div>
        </div>
    """

    # Inject the CSS and HTML into Streamlit
    st.markdown(hover_css, unsafe_allow_html=True)
    st.markdown(hover_html, unsafe_allow_html=True)
    
# Define CSS and HTML for hoverable title with image
def add_hoverable_title_with_image_inline(title_text, image_url):
    # CSS for the hover effect
    hover_css = f"""
        <style>
        .hoverable-title {{
            position: relative;
            display: inline; /* Ensure inline display */
            cursor: pointer;
            font-weight: inherit; /* Match parent font weight */
            font-size: inherit; /* Match parent font size */
            color: inherit; /* Match parent color */
        }}
        .hoverable-title .image-tooltip {{
            visibility: hidden;
            opacity: 0;
            position: absolute;
            top: 0; /* Align with the text */
            left: 90%; /* Position slightly to the right of the text */
            transform: translateX(10px); /* Add a small gap from the text */
            transition: opacity 0.3s, visibility 0.3s;
            z-index: 999;
        }}
        .hoverable-title:hover .image-tooltip {{
            visibility: visible;
            opacity: 1;
        }}
        .image-tooltip img {{
            max-width: 80px; /* Slightly smaller image size */
            background: transparent; /* Transparent background */
            border-radius: 8px;
        }}
        </style>
    """

    # HTML structure for the title with hoverable image
    hover_html = f"""
        <span class="hoverable-title">
            {title_text}
            <span class="image-tooltip">
                <img src="{image_url}" alt="Tooltip Image">
            </span>
        </span>
    """

    # Inject the CSS and HTML into Streamlit
    st.markdown(hover_css, unsafe_allow_html=True)
    return hover_html

def inject_hover_email_css():
    hover_css = """
    <style>
    .hoverable-email {
        position: relative;
        display: inline-block;
        cursor: pointer;
        font-size: 1.0rem;
        color: #1f77b4;
    }
    .hoverable-email .image-tooltip {
        visibility: hidden;
        opacity: 0;
        position: absolute;
        top: 350%;
        left: 80%;
        transform: translateX(-50%);
        transition: opacity 0.3s, visibility 0.3s;
        z-index: 999;
    }
    .hoverable-email:hover .image-tooltip {
        visibility: visible;
        opacity: 1;
    }
    .image-tooltip img {
        width: 200px;
        height: auto;
        border-radius: 10px;
        background: transparent;
    }
    </style>
    """
    st.markdown(hover_css, unsafe_allow_html=True)
        
def save_raw(session_state):
    save_path = session_state.submitted_data["save_path"]
    file_path = os.path.join(save_path, "Raw")
    if not os.path.isdir(file_path):
        os.makedirs(file_path)
    pickle_path = os.path.join(file_path, "raw_data.pkl")
    with open(pickle_path, "wb") as f:
        pickle.dump(dict(session_state), f)