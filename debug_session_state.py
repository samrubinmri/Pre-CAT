#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 12:12:53 2025

@author: jonah
"""

import pickle
import os

def load_session_state(pickle_path):
    with open(pickle_path, "rb") as f:
        session_state = pickle.load(f)
    return session_state
        
pickle_path = "/Users/jonah/Documents/MRI_Data/Berkeley/HCM_Full/20250108_165402_M1914_1_2/Data/Streamlit_Test/test_state.pkl"

session_state = load_session_state(pickle_path)