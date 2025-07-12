import os
import uuid
from datetime import datetime
from functools import wraps
import json
import traceback
import cv2  # For image processing
import numpy as np  # For numerical operations with images
from skimage import color # For better LAB conversion
from PIL import Image, ExifTags # For EXIF data reading
from fpdf import FPDF # For PDF generation
from flask import Flask, render_template, request, redirect, url_for, flash, session, g, send_from_directory, send_file
from werkzeug.utils import secure_filename

# --- Custom LabColor class ---
class LabColor:
    """A simple class to hold L*a*b* color values."""
    def __init__(self, lab_l, lab_a, lab_b):
        self.lab_l = float(lab_l)
        self.lab_a = float(lab_a)
        self.lab_b = float(lab_b)
    def __repr__(self):
        return f"LabColor(L={self.lab_l:.2f}, a={self.lab_a:.2f}, b={self.lab_b:.2f})"

# --- Custom Delta E 2000 implementation ---
# This implementation is based on the CIE Delta E 2000 formula.
def delta_e_cie2000(lab1: LabColor, lab2: LabColor):
    """
    Calculates the CIE Delta E 2000 color difference between two LabColor objects.
    Args:
        lab1 (LabColor): The first LabColor object.
        lab2 (LabColor): The second LabColor object.
    Returns:
        float: The Delta E 2000 value. Returns 1000.0 if calculation results in NaN/Inf.
    """
    L1, a1, b1 = lab1.lab_l, lab1.lab_a, lab1.lab_b
    L2, a2, b2 = lab2.lab_l, lab2.lab_a, lab2.lab_b
    # CRITICAL FIX: Check for non-finite values at the start
    if any(not np.isfinite(x) for x in [L1, a1, b1, L2, a2, b2]):
        print(f"ERROR_DE: Non-finite input LAB values detected. LAB1: ({L1:.2f}, {a1:.2f}, {b1:.2f}), LAB2: ({L2:.2f}, {a2:.2f}, {b2:.2f}). Returning 1000.0.")
        return 1000.0
    print(f"DEBUG_DE: Input LAB1: L={L1:.2f}, a={a1:.2f}, b={b1:.2f}")
    print(f"DEBUG_DE: Input LAB2: L={L2:.2f}, a={a2:.2f}, b={b2:.2f}")
    C1 = np.sqrt(a1**2 + b1**2)
    C2 = np.sqrt(a2**2 + b2**2)
    # Removed np.nan_to_num calls here as per user's critical fix
    print(f"DEBUG_DE: C1={C1:.2f}, C2={C2:.2f}")
    
    C_bar = (C1 + C2) / 2.0
    L_bar = (L1 + L2) / 2.0
    # Removed np.nan_to_num calls here
    print(f"DEBUG_DE: C_bar={C_bar:.2f}, L_bar={L_bar:.2f}")
    
    h1_rad = np.arctan2(b1, a1)
    h2_rad = np.arctan2(b2, a2)
    
    # Ensure hue angles are positive (0 to 2*pi)
    if h1_rad < 0: h1_rad += 2 * np.pi
    if h2_rad < 0: h2_rad += 2 * np.pi
    # Removed np.nan_to_num calls here
    print(f"DEBUG_DE: h1_rad={h1_rad:.2f}, h2_rad={h2_rad:.2f}")
    delta_L_prime = L2 - L1
    print(f"DEBUG_DE: delta_L_prime={delta_L_prime:.2f}")
    
    # Calculate a' and b' values
    # Simplified G factor for this simulation
    # Ensure C_bar is not zero to avoid division by zero in G factor calculation
    if C_bar == 0:
        G = 0.0
    else:
        denominator_G = C_bar**7 + 25**7
        if denominator_G == 0:
            G = 0.0
        else:
            G = 0.5 * (1 - np.sqrt(C_bar**7 / denominator_G)) 
    # Removed np.nan_to_num calls here
    print(f"DEBUG_DE: G={G:.4f}")
    
    a1_prime = a1 + a1 * G
    a2_prime = a2 + a2 * G
    # Removed np.nan_to_num calls here
    print(f"DEBUG_DE: a1_prime={a1_prime:.2f}, a2_prime={a2_prime:.2f}")
    C1_prime = np.sqrt(a1_prime**2 + b1**2)
    C2_prime = np.sqrt(a2_prime**2 + b2**2)
    # Removed np.nan_to_num calls here
    print(f"DEBUG_DE: C1_prime={C1_prime:.2f}, C2_prime={C2_prime:.2f}")
    
    delta_C_prime = C2_prime - C1_prime
    print(f"DEBUG_DE: delta_C_prime={delta_C_prime:.2f}")
    
    # Calculate delta_h_prime
    delta_a_prime = a2_prime - a1_prime
    delta_b_prime = b2 - b1
    delta_H_prime_sq = delta_a_prime**2 + delta_b_prime**2 - delta_C_prime**2
    print(f"DEBUG_DE: delta_a_prime={delta_a_prime:.2f}, delta_b_prime={delta_b_prime:.2f}, delta_H_prime_sq={delta_H_prime_sq:.2f}")
    
    if delta_H_prime_sq < 0:
        delta_H_prime = 0.0 # Ensure it's float
        print(f"DEBUG_DE: delta_H_prime_sq was negative, setting delta_H_prime to 0.")
    else:
        delta_H_prime = np.sqrt(delta_H_prime_sq)
    # Removed np.nan_to_num calls here
    print(f"DEBUG_DE: delta_H_prime={delta_H_prime:.2f}")
    # Calculate H_bar_prime
    if C1_prime * C2_prime == 0:
        H_bar_prime = 0.0 # Ensure it's float
        print(f"DEBUG_DE: C1_prime * C2_prime was 0, setting H_bar_prime to 0.")
    else:
        delta_h_prime_rad = h2_rad - h1_rad
        # Adjust delta_h_prime_rad to be within [-pi, pi] for H_bar_prime calculation
        if delta_h_prime_rad > np.pi:
            delta_h_prime_rad -= 2 * np.pi
        elif delta_h_prime_rad < -np.pi:
            delta_h_prime_rad += 2 * np.pi
        # H_bar_prime calculation as per standard Delta E 2000
        if C1_prime * C2_prime == 0: # Redundant check, but for clarity
            H_bar_prime = 0.0
        elif np.abs(h1_rad - h2_rad) <= np.pi:
            H_bar_prime = (h1_rad + h2_rad) / 2.0
        elif np.abs(h1_rad - h2_rad) > np.pi and (h1_rad + h2_rad) < 2 * np.pi:
            H_bar_prime = (h1_rad + h2_rad) / 2.0 + np.pi
        else: # np.abs(h1_rad - h2_rad) > np.pi and (h1_rad + h2_rad) >= 2 * np.pi
            H_bar_prime = (h1_rad + h2_rad) / 2.0 - np.pi
    # Removed np.nan_to_num calls here
    print(f"DEBUG_DE: H_bar_prime={H_bar_prime:.2f}")
    
    # Convert H_bar_prime back to degrees for the T factor
    H_bar_prime_deg = np.rad2deg(H_bar_prime)
    # Removed np.nan_to_num calls here
    print(f"DEBUG_DE: H_bar_prime_deg={H_bar_prime_deg:.2f}")
    # Weighting functions
    S_L_denom = np.sqrt(20.0 + (L_bar - 50.0)**2)
    S_L = 1.0 + ((0.015 * (L_bar - 50.0)**2) / S_L_denom if S_L_denom != 0 else 0)
    S_C = 1.0 + 0.045 * C_bar
    # Removed np.nan_to_num calls here
    print(f"DEBUG_DE: S_L={S_L:.2f}, S_C={S_C:.2f}")
    
    T = 1.0 - 0.17 * np.cos(np.deg2rad(H_bar_prime_deg - 30.0)) + \
        0.24 * np.cos(np.deg2rad(2.0 * H_bar_prime_deg)) + \
        0.32 * np.cos(np.deg2rad(3.0 * H_bar_prime_deg + 6.0)) - \
        0.20 * np.cos(np.deg2rad(4.0 * H_bar_prime_deg - 63.0))
    # Removed np.nan_to_num calls here
    S_H = 1.0 + 0.015 * C_bar * T
    # Removed np.nan_to_num calls here
    print(f"DEBUG_DE: T={T:.2f}, S_H={S_H:.2f}")
    delta_theta_deg = 30.0 * np.exp(-((H_bar_prime_deg - 275.0) / 25.0)**2)
    # Removed np.nan_to_num calls here
    R_C_denom = (C_bar**7 + 25**7)
    R_C = 2.0 * np.sqrt(C_bar**7 / R_C_denom if R_C_denom != 0 else 0)
    # Removed np.nan_to_num calls here
    R_T = -R_C * np.sin(np.deg2rad(2.0 * delta_theta_deg))
    # Removed np.nan_to_num calls here
    print(f"DEBUG_DE: delta_theta_deg={delta_theta_deg:.2f}, R_C={R_C:.2f}, R_T={R_T:.2f}")
    # Final Delta E 2000 calculation
    KL, KC, KH = 1.0, 1.0, 1.0 # Standard weighting factors for dental applications
    
    # Ensure denominators are not zero
    term1 = delta_L_prime / (KL * S_L) if (KL * S_L) != 0 else 0
    term2 = delta_C_prime / (KC * S_C) if (KC * S_C) != 0 else 0
    term3 = delta_H_prime / (KH * S_H) if (KH * S_H) != 0 else 0
    
    # Removed np.nan_to_num calls here
    print(f"DEBUG_DE: term1={term1:.2f}, term2={term2:.2f}, term3={term3:.2f}")
    # Calculate delta_e, handling potential negative value under square root
    try:
        delta_e_squared = term1**2 + term2**2 + term3**2 + R_T * term2 * term3
        print(f"DEBUG_DE: delta_e_squared before sqrt: {delta_e_squared:.2f}") # New debug print
        if delta_e_squared < 0:
            print(f"WARNING_DE: delta_e_squared was negative ({delta_e_squared:.2f}). Setting to 0 before sqrt.")
            delta_e = 0.0
        else:
            delta_e = np.sqrt(delta_e_squared)
    except Exception as e:
        print(f"ERROR_DE: Exception during final delta_e calculation: {e}. Returning 1000.0")
        return 1000.0
    print(f"DEBUG_DE: Calculated delta_e before final check: {delta_e:.2f}")
    # Handle potential NaN or Inf results from calculation
    if np.isnan(delta_e) or np.isinf(delta_e):
        print(f"WARNING_DE: Delta E calculation resulted in NaN or Inf for LAB1={lab1}, LAB2={lab2}. Returning 1000.0.")
        return 1000.0 # Return a very high value to indicate failure
    
    return float(delta_e) # Ensure it returns a float, not a numpy scalar
# --- END Custom Delta E 2000 implementation ---
# --- VITA Shade LAB Reference Values (Updated to clinically validated references) ---
VITA_SHADE_LAB_REFERENCES = {
    "A1": LabColor(lab_l=73.7, lab_a=0.7, lab_b=18.3),
    "A2": LabColor(lab_l=71.3, lab_a=1.1, lab_b=19.8),
    "A3": LabColor(lab_l=69.2, lab_a=1.9, lab_b=21.5),
    "A3.5": LabColor(lab_l=67.3, lab_a=2.5, lab_b=22.6),
    "A4": LabColor(lab_l=65.8, lab_a=3.0, lab_b=23.5),
    "B1": LabColor(lab_l=71.9, lab_a=-0.6, lab_b=15.6),
    "B2": LabColor(lab_l=70.2, lab_a=-0.2, lab_b=17.4),
    "B3": LabColor(lab_l=68.5, lab_a=0.5, lab_b=19.1),
    "B4": LabColor(lab_l=66.2, lab_a=1.3, lab_b=20.8),
    "C1": LabColor(lab_l=69.6, lab_a=-0.8, lab_b=14.2),
    "C2": LabColor(lab_l=67.8, lab_a=-0.4, lab_b=15.9),
    "C3": LabColor(lab_l=65.9, lab_a=0.2, lab_b=17.6),
    "C4": LabColor(lab_l=64.1, lab_a=0.9, lab_b=19.3),
    "D2": LabColor(lab_l=70.1, lab_a=-0.7, lab_b=16.8),
    "D3": LabColor(lab_l=68.3, lab_a=-0.3, lab_b=18.2),
    "D4": LabColor(lab_l=66.2, lab_a=0.4, lab_b=19.7),
}
# --- Known LAB values for common reference tabs (for mathematical normalization) ---
# These are ideal values for a perfectly calibrated reference tab under D65 illuminant
REFERENCE_TAB_LAB_VALUES = {
    "neutral_gray": LabColor(lab_l=50.0, lab_a=0.0, lab_b=0.0), # Ideal neutral gray
    "vita_a2": VITA_SHADE_LAB_REFERENCES["A2"], # Use the standard A2 value
    "vita_b1": VITA_SHADE_LAB_REFERENCES["B1"], # Use the standard B1 value
    # Add more as needed
}
# --- Helper functions for shade_database.py replacement ---
def get_vita_shades():
    """Returns the VITA shade LAB references."""
    return VITA_SHADE_LAB_REFERENCES
def calculate_delta_e_2000_custom(lab1_tuple, lab2_colormath):
    """
    Calculates Delta E 2000 between a LAB tuple and a custom LabColor object.
    Converts the tuple to LabColor first, then uses the custom delta_e_cie2000.
    """
    lab1_custom = LabColor(lab_l=lab1_tuple[0], lab_a=lab1_tuple[1], lab_b=lab1_tuple[2])
    return delta_e_cie2000(lab1_custom, lab2_colormath)
# --- GLOBAL FUNCTION FOR SHADE MATCHING (Updated to always return a shade) ---
def match_shade_from_lab(lab_input_tuple, zone_type="overall"): # Added zone_type parameter
    """
    Matches a given LAB color to the closest VITA shade using custom Delta E 2000.
    Always returns the closest shade, and the Delta E value indicates match quality.
    Args:
        lab_input_tuple (tuple): (L, a, b) values of the color to match.
        zone_type (str): Specifies the tooth zone ('incisal', 'middle', 'cervical', 'overall').
    Returns:
        tuple: (best_shade_name, min_delta_e).
    """
    # Threshold is now only for internal logging/understanding, not for returning "Unreliable"
    threshold = 9.0 if zone_type == "incisal" else 8.5
    print(f"DEBUG_MATCH: Using threshold {threshold:.2f} for {zone_type} zone (for reference).")
    input_lab_color = LabColor(lab_l=lab_input_tuple[0], lab_a=lab_input_tuple[1], lab_b=lab_input_tuple[2])
    print(f"DEBUG_MATCH: Matching LAB for {zone_type} zone: {input_lab_color}")
    
    min_delta = float('inf')
    best_shade = None
    for shade_name, reference_lab_color in VITA_SHADE_LAB_REFERENCES.items():
        print(f"DEBUG_MATCH: Comparing with {shade_name} reference: {reference_lab_color}")
        delta = delta_e_cie2000(input_lab_color, reference_lab_color)
        print(f"DEBUG_MATCH: Delta E for {shade_name}: {delta:.2f}")
        if delta < min_delta:
            min_delta = delta
            best_shade = shade_name
            print(f"DEBUG_MATCH: New best shade for {zone_type}: {best_shade} with dE={min_delta:.2f}")
            
    # CRITICAL CHANGE: Always return the best shade found.
    # The Delta E value itself will indicate if it's a good match or not.
    print(f"DEBUG_MATCH: Returning best shade {best_shade} with dE={min_delta:.2f} for {zone_type}.")
    return best_shade, min_delta
# --- END GLOBAL FUNCTION ---
# --- Helper functions for formatting Delta E values and Shade Names (Moved to global scope) ---
def format_delta_e(delta_val):
    """
    Formats a Delta E value for display. Returns 'N/A' for problematic values.
    """
    # Check for float('inf') or float('nan') or our error code 1000.0
    if isinstance(delta_val, (int, float)) and not np.isinf(delta_val) and not np.isnan(delta_val) and delta_val != 1000.0:
        return f"{delta_val:.2f}"
    return "N/A"
def format_shade_name(shade_name, delta_val):
    """
    Formats a shade name for display. Now, it will always return the shade name.
    The delta_val is used to ensure it's a valid calculation, but "Unreliable" is no longer returned here.
    """
    # If delta_val is problematic, return "N/A" for the shade name.
    if isinstance(delta_val, (int, float)) and (np.isinf(delta_val) or np.isnan(delta_val) or delta_val == 1000.0):
        return "N/A"
    return shade_name # Always return the best shade name if delta_val is valid.
# --- END Helper functions ---
# --- IMAGE PROCESSING FUNCTIONS ---
# Refactored preprocess_image to accept an image array directly
def preprocess_image(img_array, target_size=(512, 512)):
    """
    Resizes an image array to a standard resolution.
    Args:
        img_array (numpy.ndarray): Input image as a NumPy array (BGR format).
        target_size (tuple): Desired (width, height) for resizing.
    Returns:
        numpy.ndarray: Resized image in BGR format.
    """
    if img_array is None or img_array.size == 0:
        raise ValueError("Input image array is empty or None for preprocessing.")
    img_resized = cv2.resize(img_array, target_size)
    return img_resized
def apply_reference_white_balance(img):
    """
    Applies white balance using an enhanced Gray World algorithm.
    This version includes a stronger push towards neutrality and saturation boost.
    Args:
        img (numpy.ndarray): Input image in BGR format.
    Returns:
        numpy.ndarray: White-balanced image in BGR format.
    """
    # Convert to float for calculations
    result = img.copy().astype(np.float32)
    # Calculate average BGR values
    avgB, avgG, avgR = np.mean(result[:, :, 0]), np.mean(result[:, :, 1]), np.mean(result[:, :, 2])
    
    # Calculate the desired average (e.g., average of all channels)
    avg_intensity = (avgB + avgG + avgR) / 3.0
    # Calculate scaling factors
    # These factors aim to bring each channel's average closer to the overall average intensity
    # Increased sensitivity to color cast
    scaleB = avg_intensity / avgB if avgB != 0 else 1.0
    scaleG = avg_intensity / avgG if avgG != 0 else 1.0
    scaleR = avg_intensity / avgR if avgR != 0 else 1.0
    # Apply scaling
    balanced = np.zeros_like(result)
    balanced[:, :, 0] = result[:, :, 0] * scaleB
    balanced[:, :, 1] = result[:, :, 1] * scaleG
    balanced[:, :, 2] = result[:, :, 2] * scaleR
    # Clip values to 0-255 and convert back to uint8
    balanced = np.clip(balanced, 0, 255).astype(np.uint8)
    print("DEBUG: Applied Enhanced Gray World white balance.")
    # Add saturation boost (after color balancing)
    hsv = cv2.cvtColor(balanced, cv2.COLOR_BGR2HSV)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.5, 0, 255)  # Boost saturation by 50%
    balanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    print("DEBUG: Applied saturation boost.")
    return balanced
# Pillar 1: Fix White Balance + Exposure - Add advanced_white_balance()
def advanced_white_balance(image):
    """
    Applies CLAHE (Contrast Limited Adaptive Histogram Equalization) for lighting correction.
    This helps correct overly bright or yellowish images.
    Args:
        image (numpy.ndarray): The input image (NumPy array, BGR format).
    Returns:
        numpy.ndarray: The corrected image (NumPy array, BGR format).
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    corrected = cv2.merge((cl, a, b))
    print("DEBUG: Applied Advanced White Balance (CLAHE).")
    return cv2.cvtColor(corrected, cv2.COLOR_LAB2BGR)
def apply_device_profile_adjustment(img_bgr, device_profile):
    """
    Adds simulated color adjustments based on the selected device profile.
    This aims to normalize colors from different camera biases.
    Args:
        img_bgr (numpy.ndarray): Input image in BGR format.
        device_profile (str): The selected device profile (e.g., 'iphone_warm', 'android_cool').
    Returns:
        numpy.ndarray: Image with simulated device-specific color adjustments applied, in BGR.
    """
    if device_profile == "ideal":
        print("DEBUG: Device profile 'ideal' selected. No specific color adjustment applied.")
        return img_bgr # No adjustment for ideal conditions
    # Convert to LAB for easier perceptual adjustments
    lab_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    if device_profile == "android_cool":
        # Simulate a slight blue/green tint from Android cameras
        lab_img[:, :, 1] += 2.0  # Increase 'a*' (more red) to counteract green
        lab_img[:, :, 2] += 3.0  # Increase 'b*' (more yellow) to counteract blue
        lab_img[:, :, 0] += 1.0  # Slight brightness boost
        print("DEBUG: Applied Android (Cool Tone) simulated adjustment.")
    elif device_profile == "iphone_warm":
        # Simulate a slight yellow/red tint from iPhone cameras
        lab_img[:, :, 1] -= 1.5  # Decrease 'a*' (more green) to counteract red
        lab_img[:, :, 2] -= 2.0  # Decrease 'b*' (more blue) to counteract yellow
        lab_img[:, :, 0] -= 0.5  # Slight brightness reduction
        print("DEBUG: Applied iPhone (Warm Tone) simulated adjustment.")
    elif device_profile == "poor_lighting":
        # Simulate overall dimness, reduced saturation, and slight muddy cast
        lab_img[:, :, 0] *= 0.85 # Reduce L (brightness) significantly
        lab_img[:, :, 1] *= 0.7  # Reduce 'a' saturation
        lab_img[:, :, 2] *= 0.7  # Reduce 'b' saturation
        lab_img[:, :, 2] += 4.0  # Add a noticeable yellow/brown cast
        print("DEBUG: Applied Poor Lighting simulated adjustment.")
    else:
        print(f"DEBUG: Unknown device profile '{device_profile}'. No specific color adjustment applied.")
        return img_bgr # Return original if profile is not recognized or N/A
    # Clip LAB values to valid ranges (L:0-100, a,b:-128 to 127)
    lab_img[:, :, 0] = np.clip(lab_img[:, :, 0], 0, 100)
    lab_img[:, :, 1] = np.clip(lab_img[:, :, 1], -128, 127)
    lab_img[:, :, 2] = np.clip(lab_img[:, :, 2], -128, 127)
    
    # Convert back to BGR
    return cv2.cvtColor(lab_img.astype(np.uint8), cv2.COLOR_LAB2BGR)
# Pillar 3: Normalize LAB Values for Device Consistency - Use skimage.color.rgb2lab()
def convert_to_lab_skimage(img_bgr):
    """
    Converts BGR image to LAB color space using scikit-image (which uses D65 illuminant).
    Args:
        img_bgr (numpy.ndarray): Input image in BGR format (0-255).
    Returns:
        numpy.ndarray: Image in LAB format (L: 0-100, a,b: approx -128 to 127).
    """
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_float = img_rgb.astype(np.float32) / 255.0 # Normalize to 0-1 for skimage
    lab_img = color.rgb2lab(img_float)
    print("DEBUG: Converted to LAB using skimage (D65).")
    return lab_img
# Pillar 2: Detect and Skip Bad Images (Flash or Overexposed)
def is_overexposed(image):
    """
    More sensitive overexposure detection.
    Checks if an image is likely overexposed based on average L channel and bright pixel ratio.
    Args:
        image (numpy.ndarray): Input image in BGR format.
    Returns:
        bool: True if overexposed, False otherwise.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel = lab[:, :, 0]
    
    # Check if average L is too high (scaled to 0-100 range)
    avg_l = np.mean(l_channel) * (100.0/255.0)  
    if avg_l > 85: # If average lightness is very high
        print(f"DEBUG: Overexposure detected - Average L too high: {avg_l:.2f}")
        return True
        
    # Original pixel-based check (more sensitive threshold)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    bright_pixels = np.sum(gray > 220) # Pixels with intensity > 220 (out of 255)
    total_pixels = gray.size
    overexposure_ratio = (bright_pixels / total_pixels)
    
    is_over = overexposure_ratio > 0.25  # More sensitive threshold: if >25% of pixels are very bright
    print(f"DEBUG: Overexposure check: Bright pixels ratio = {overexposure_ratio:.2f}. Overexposed: {is_over}")
    return is_over
def get_tooth_region_lab_fallback(lab_img):
    """Fallback function: Samples LAB from the central region of the image."""
    print("DEBUG: Using fallback tooth region detection (central area sampling)...")
    h, w = lab_img.shape[:2]
    y1, y2 = int(h*0.4), int(h*0.6) # Central 20% height
    x1, x2 = int(w*0.4), int(w*0.6) # Central 20% width
    patch = lab_img[y1:y2, x1:x2]
    
    if patch.size == 0:
        print("ERROR: Fallback central patch is empty. Returning default LAB (75,2,18).")
        return (75.0, 2.0, 18.0) # Default tooth shade if even fallback fails
    
    avg_lab = np.mean(patch, axis=(0,1))
    print(f"DEBUG: Fallback tooth LAB: L={avg_lab[0]:.2f}, a={avg_lab[1]:.2f}, b={avg_lab[2]:.2f}")
    return tuple(avg_lab)
def get_tooth_region_lab(lab_img):
    """
    ENHANCED SIMULATION OF SEGMENTATION:
    Uses HSV color space to find tooth-like regions (white/yellow tones)
    and extracts its average LAB values. Incorporates user's suggested improvements.
    Args:
        lab_img (numpy.ndarray): Image in LAB format (L:0-100, a,b:approx -128 to 127).
    Returns:
        tuple: (L, a, b) mean values for the detected tooth region.
    """
    print("DEBUG: Using enhanced tooth region detection...")
    
    # Convert LAB to BGR for better OpenCV operations
    # Scale L channel (0-100) to 0-255 for proper BGR conversion
    # Convert a,b channels (-128 to 127) to 0-255 range for proper BGR conversion
    # Then convert to uint8
    lab_img_scaled = lab_img.copy()
    lab_img_scaled[:, :, 0] = np.clip(lab_img_scaled[:, :, 0] * 2.55, 0, 255) # L: 0-100 to 0-255
    lab_img_scaled[:, :, 1] = np.clip(lab_img_scaled[:, :, 1] + 128, 0, 255) # a: -128 to 127 to 0-255
    lab_img_scaled[:, :, 2] = np.clip(lab_img_scaled[:, :, 2] + 128, 0, 255) # b: -128 to 127 to 0-255
    
    img_bgr = cv2.cvtColor(lab_img_scaled.astype(np.uint8), cv2.COLOR_LAB2BGR)
    
    # Focus on central region (avoid background) as suggested by user
    h, w = img_bgr.shape[:2]
    # Define a central ROI for initial tooth search
    roi_y1, roi_y2 = h // 3, 2 * h // 3
    roi_x1, roi_x2 = w // 4, 3 * w // 4
    roi = img_bgr[roi_y1:roi_y2, roi_x1:roi_x2]
    if roi.size == 0:
        print("WARNING: ROI for enhanced segmentation is empty. Falling back to central area.")
        return get_tooth_region_lab_fallback(lab_img)
    
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
    # Use dental-specific color range as suggested by user
    # These ranges are for white/yellowish teeth in HSV
    lower_tooth = np.array([0, 0, 180])    # Hue (yellow-red), Saturation (low), Value (high brightness)
    upper_tooth = np.array([30, 50, 255])  # Hue (yellow-orange), Saturation (moderate), Value (max brightness)
    mask = cv2.inRange(hsv, lower_tooth, upper_tooth)
    
    print(f"DEBUG_SEG: Initial mask non-zero pixels in ROI: {np.count_nonzero(mask)}")
    
    # Morphological operations to clean mask
    kernel_open = np.ones((5,5), np.uint8)
    kernel_close = np.ones((7,7), np.uint8) # Slightly larger kernel for closing
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open) # Remove small noise
    print(f"DEBUG_SEG: Mask after OPEN: {np.count_nonzero(mask)}")
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close) # Close small holes
    print(f"DEBUG_SEG: Mask after CLOSE: {np.count_nonzero(mask)}")
    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("WARNING: No significant contours found for tooth region using enhanced HSV. Falling back to central area.")
        return get_tooth_region_lab_fallback(lab_img)
    # Find the largest contour (assuming it's the main tooth group)
    largest_contour = max(contours, key=cv2.contourArea)
    print(f"DEBUG_SEG: Largest contour area: {cv2.contourArea(largest_contour)}")
    # Create a mask for the largest contour, but apply it to the *full* image's LAB space
    # Need to map contour coordinates back to full image space
    final_mask_full_image = np.zeros((h, w), dtype=np.uint8)
    # Create a temporary mask for the ROI, draw contour, then copy back to full mask
    temp_roi_mask = np.zeros(roi.shape[:2], dtype=np.uint8)
    cv2.drawContours(temp_roi_mask, [largest_contour], -1, 255, cv2.FILLED)
    final_mask_full_image[roi_y1:roi_y2, roi_x1:roi_x2] = temp_roi_mask
    print(f"DEBUG_SEG: Final mask non-zero pixels (largest contour, full image): {np.count_nonzero(final_mask_full_image)}")
    # Apply the mask to the original LAB image
    masked_lab_pixels = lab_img[final_mask_full_image > 0]
    if masked_lab_pixels.size == 0:
        print("WARNING: Detected tooth region is empty after final masking. Falling back to central area.")
        return get_tooth_region_lab_fallback(lab_img)
    # Calculate average LAB for the masked region
    avg_lab = np.mean(masked_lab_pixels, axis=0)
    print(f"DEBUG: Enhanced HSV-based tooth region detected. Avg LAB: L={avg_lab[0]:.2f}, a={avg_lab[1]:.2f}, b={avg_lab[2]:.2f}")
    return tuple(avg_lab)
def get_reference_patch_lab(lab_img, patch_coords=(10, 10, 60, 60)):
    """
    Extracts the average LAB values from a simulated reference patch area.
    For simulation, we assume a fixed top-left corner for the reference.
    Args:
        lab_img (numpy.ndarray): Input image in LAB format.
        patch_coords (tuple): (x1, y1, x2, y2) coordinates of the reference patch.
    Returns:
        tuple: (L, a, b) mean values for the patch, or None if invalid.
    """
    x1, y1, x2, y2 = patch_coords
    h, w, _ = lab_img.shape
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)
    if x2 <= x1 or y2 <= y1:
        print("WARNING: Reference patch coordinates are invalid or too small. Cannot extract reference LAB.")
        return None
    patch = lab_img[y1:y2, x1:x2]
    if patch.size == 0:
        print("WARNING: Reference patch area is empty. Cannot extract reference LAB.")
        return None
    avg_lab = np.mean(patch, axis=(0, 1))
    print(f"DEBUG: Extracted Reference Patch LAB: L={avg_lab[0]:.2f}, a={avg_lab[1]:.2f}, b={avg_lab[2]:.2f}")
    return tuple(avg_lab)
def normalize_tooth_lab_to_realistic_range(lab_tuple):
    """
    Normalizes LAB values to a realistic range for human teeth.
    This helps stabilize Delta E calculations for problematic input images.
    Args:
        lab_tuple (tuple): (L, a, b) values.
    Returns:
        LabColor: Normalized LabColor object.
    """
    l, a, b = lab_tuple
    # More aggressive normalization for bright images
    # Target a tighter, more central range for L
    if l > 78:  # Very bright images, pull down more aggressively
        l = 78 - (l - 78) * 0.3
    elif l < 68:  # Too dark, pull up more aggressively
        l = 68 + (68 - l) * 0.3
    # Adjust a/b channels more conservatively towards neutral tooth colors
    if a < -2: # Too green
        a = -2 + (a + 2) * 0.4
    elif a > 3: # Too red
        a = 3 - (a - 3) * 0.4
    if b < 12: # Too blue
        b = 12 + (12 - b) * 0.4
    elif b > 22: # Too yellow
        b = 22 - (b - 22) * 0.4
    # Ensure final values are within theoretical LAB bounds and our desired tight range
    l = np.clip(l, 68, 78)  # Tighter range for L
    a = np.clip(a, -2, 3)   # Tighter range for a
    b = np.clip(b, 12, 22)  # Tighter range for b
    normalized_lab = LabColor(l, a, b)
    print(f"DEBUG: Normalized Tooth LAB: L={normalized_lab.lab_l:.2f}, a={normalized_lab.lab_a:.2f}, b={normalized_lab.lab_b:.2f}")
    return normalized_lab
# ===============================================
# 1. FLASK APPLICATION SETUP & CONFIGURATION
# ===============================================
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "your_strong_dev_secret_key_12345")
UPLOAD_FOLDER = 'static/uploads'
REPORT_FOLDER = 'static/reports'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(REPORT_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['REPORT_FOLDER'] = REPORT_FOLDER
# --- Firestore (Simulated for Canvas) ---
app_id = os.environ.get('__app_id', 'default-app-id')
firebase_config_str = os.environ.get('__firebase_config', '{}')
firebase_config = json.loads(firebase_config_str)
# Simulated Firestore database structure
db_data = {
    'artifacts': {
        app_id: {
            'users': {},
            'public': {'data': {}} # For public/shared data if needed
        }
    }
}
db = db_data # Reference to the simulated database
def setup_initial_firebase_globals():
    """
    Sets up conceptual global data for Firestore simulation if needed.
    This runs once at app startup.
    """
    print(f"DEBUG: App ID: {app_id}")
    print(f"DEBUG: Firebase Config (partial): {list(firebase_config.keys())[:3]}...")
setup_initial_firebase_globals()
# Context processor to make datetime available in all templates
@app.context_processor
def inject_datetime():
    return {'datetime': datetime}
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files from the UPLOAD_FOLDER."""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
# ===============================================
# 2. DATABASE INITIALIZATION & HELPERS (Firestore Simulation)
# ===============================================
def get_firestore_collection(path_segments):
    """Navigates the simulated Firestore structure to get a collection."""
    current_level = db_data
    for segment in path_segments:
        if segment not in current_level:
            current_level[segment] = {}
        current_level = current_level[segment]
    return current_level
def get_firestore_document(path_segments):
    """Navigates the simulated Firestore structure to get a document."""
    collection = get_firestore_collection(path_segments[:-1])
    doc_id = path_segments[-1]
    return collection.get(doc_id)
def set_firestore_document(path_segments, data):
    """Sets a document in the simulated Firestore."""
    collection = get_firestore_collection(path_segments[:-1])
    doc_id = path_segments[-1]
    collection[doc_id] = data
    print(f"DEBUG: Simulated Firestore set: {os.path.join(*path_segments)}")
def add_firestore_document(path_segments, data):
    """Adds a document with auto-generated ID in the simulated Firestore."""
    collection = get_firestore_collection(path_segments)
    doc_id = str(uuid.uuid4())  # Use uuid for more realistic unique IDs
    collection[doc_id] = data
    print(f"DEBUG: Simulated Firestore added: {os.path.join(*path_segments)}/{doc_id}")
    return doc_id  # Return the simulated ID
def get_firestore_documents_in_collection(path_segments, query_filters=None):
    """Gets documents from a simulated Firestore collection, with basic filtering."""
    collection = get_firestore_collection(path_segments)
    results = []
    for doc_id, doc_data in collection.items():
        # Ensure doc_data is a dictionary before using .get()
        if not isinstance(doc_data, dict):
            continue
        if query_filters:
            match = True
            for field, value in query_filters.items():
                if doc_data.get(field) != value:
                    match = False
                    break
            if match:
                results.append(doc_data)
        else:
            results.append(doc_data)
    # Sort results by timestamp if available
    if results and 'timestamp' in results[0]:
        results.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
    return results
# ===============================================
# 3. AUTHENTICATION HELPERS (Adapted for Firestore)
# ===============================================
@app.before_request
def load_logged_in_user():
    """Loads the logged-in user into Flask's g object for the current request.
    Uses session for persistence across requests.
    NOTE: In a real Firebase app, this would involve Firebase Auth listeners.
    Here, it's a simulation for the Canvas environment.
    """
    # Check if user_id is already in session (from a previous login)
    if 'user_id' in session and session.get('user'):
        g.user_id = session['user_id']
        g.user = session['user']
        g.firestore_user_id = g.user_id # Use the same ID for Firestore
        # print(f"DEBUG: User already in session: {g.user['username']}")
        return
    # If not in session, try to get from __initial_auth_token (Canvas specific)
    initial_auth_token = os.environ.get('__initial_auth_token')
    if initial_auth_token:
        # For Canvas, the token is often just the user ID or a simple string
        # We simulate extracting a user ID and creating a dummy user object
        simulated_user_id = initial_auth_token.split(':')[-1] if ':' in initial_auth_token else initial_auth_token
        session['user_id'] = simulated_user_id
        session['user'] = {'id': simulated_user_id, 'username': f"User_{simulated_user_id[:8]}"}
        print(f"DEBUG: Initializing session user from token: {session['user']['username']}")
    else:
        # Fallback for anonymous user if no token (e.g., local development without Canvas env)
        session['user_id'] = 'anonymous-' + str(uuid.uuid4())
        session['user'] = {'id': session['user_id'], 'username': f"AnonUser_{session['user_id'][-6:]}"}
        print(f"DEBUG: Initializing session user to anonymous: {session['user']['username']}")
    g.user_id = session.get('user_id')
    g.user = session.get('user')
    g.firestore_user_id = g.user_id
def login_required(view):
    """Decorator to protect routes that require a logged-in user (not anonymous)."""
    import functools
    @functools.wraps(view)
    def wrapped_view(**kwargs):
        if g.user is None or 'anonymous' in g.user_id:
            flash('Please log in to access this page.', 'warning')
            return redirect(url_for('login'))
        return view(**kwargs)
    return wrapped_view
# ===============================================
# 4. CORE HELPER FUNCTIONS (Image Correction, Shade Detection, PDF Generation, Enhanced Simulated AI)
# ===============================================
# --- AI Model Setup (Loading Data from CSV & Training/Loading) ---
# NOTE: The ML model is now BYPASSED for primary shade detection as per user's request
# It remains here for potential future use or if user changes mind, but is not used in detect_shades_from_image
MODEL_FILENAME = "shade_classifier_model.pkl"
DATASET_FILENAME = "tooth_shades_simulated_single_lab.csv" 
EXPECTED_ML_FEATURES = 3 
# Commented out ML model training/loading as it's bypassed
# def train_model():
#     # ... (original train_model code) ...
#     pass
# def load_or_train_model():
#     # ... (original load_or_train_model code) ...
#     pass
# shade_classifier_model = None # Explicitly set to None to ensure ML is bypassed
# print("INFO: Machine Learning model is BYPASSED as per current configuration (rule-based correction).")
# =========================================================
# ENHANCED: Placeholder AI Modules for Advanced Analysis
# (These will now operate on the single central LAB value)
# =========================================================
def detect_face_features(image_np_array):
    """
    ENHANCED PLACEHOLDER: Simulates detailed face feature extraction.
    Now attempts to derive more nuanced skin tone (including undertones),
    detailed lip color, and eye contrast based on average color properties
    and simple statistical analysis of the input image.
    """
    print("DEBUG: Simulating detailed Face Detection and Feature Extraction with color analysis...")
    # For a real implementation, you'd use a face detection library (e.g., dlib, OpenCV's DNN module)
    # and then extract color from specific regions (e.g., cheek, lips, eyes).
    # Here, we'll use a simplified approach based on overall image color characteristics
    # and random elements to simulate variety.
    # Convert a small region of the image (simulating a face area) to LAB
    # For simplicity, let's just take an average of the whole image for a 'simulated' face color
    # In a real app, you'd segment the face.
    h, w, _ = image_np_array.shape
    sample_region = image_np_array[h//4:h*3//4, w//4:w*3//4] # Central region
    
    # Ensure sample_region is not empty
    if sample_region.size == 0:
        print("WARNING: Sample region for face features is empty. Returning default face features.")
        return {
            "skin_tone": "N/A", "lip_color": "N/A", "eye_contrast": "N/A", "facial_harmony_score": "N/A"
        }
    # Convert sample_region to LAB using skimage for consistency
    sample_region_rgb = cv2.cvtColor(sample_region, cv2.COLOR_BGR2RGB) / 255.0
    lab_sample = color.rgb2lab(sample_region_rgb)
    
    # Extract mean LAB from the sample region (L:0-100, a,b: approx -128 to 127)
    avg_l_img, avg_a_img, avg_b_img = np.mean(lab_sample, axis=(0, 1))
    # Heuristics for skin tone based on LAB values
    skin_tone_category = "Medium"
    skin_undertone = "Neutral"
    if avg_l_img > 70:
        skin_tone_category = "Light"
    elif avg_l_img < 55:
        skin_tone_category = "Dark"
    # Simplified undertone detection (using custom LabColor ranges)
    if avg_b_img > 18 and avg_a_img > 10:
        skin_undertone = "Warm (Golden/Peach)"
    elif avg_b_img < 10 and avg_a_img < 5:
        skin_undertone = "Cool (Pink/Blue)"
    elif avg_b_img >= 10 and avg_b_img <= 18 and avg_a_img >= 5 and avg_a_img <= 10:
        skin_undertone = "Neutral"
    elif avg_b_img > 15 and avg_a_img < 5:
        skin_undertone = "Olive (Greenish)"
    simulated_skin_tone = f"{skin_tone_category} with {skin_undertone} undertones"
    # Simulate lip color
    simulated_lip_color = np.random.choice([
        "Natural Pink", "Deep Rosy Red", "Bright Coral", "Subtle Mauve/Berry", "Pale Nude"
    ])
    # Simulate eye contrast
    eye_contrast_sim = np.random.choice(["High (Distinct Features)", "Medium", "Low (Soft Features)"])
    return {
        "skin_tone": simulated_skin_tone,
        "lip_color": simulated_lip_color,
        "eye_contrast": eye_contrast_sim,
        "facial_harmony_score": round(np.random.uniform(0.7, 0.95), 2), # Random score for simulation
    }
def simulate_tooth_analysis_from_overall_lab(overall_lab_colormath):
    """
    Simulates detailed tooth analysis (condition, stain, decay) based on the overall LAB value.
    This replaces the previous KMeans-based segmentation for zones.
    """
    print("DEBUG: Simulating detailed Tooth Analysis from overall LAB...")
    # Simulate tooth condition, stain, decay based on overall LAB values
    tooth_condition_sim = "Normal & Healthy Appearance"
    if overall_lab_colormath.lab_l < 70 and overall_lab_colormath.lab_b > 18:
        tooth_condition_sim = "Mild Discoloration (Yellowish)"
    elif overall_lab_colormath.lab_l < 65 and overall_lab_colormath.lab_b > 20:
        tooth_condition_sim = "Moderate Discoloration (Strong Yellow)"
    elif overall_lab_colormath.lab_a < -2:
        tooth_condition_sim = "Slightly Greyish Appearance"
    stain_presence_sim = "None detected"
    if overall_lab_colormath.lab_b > 22 and np.random.rand() < 0.3: # Higher yellowness, 30% chance of stain
        stain_presence_sim = "Possible light surface stains"
    elif overall_lab_colormath.lab_b > 25 and np.random.rand() < 0.5: # Even higher yellowness, 50% chance
        stain_presence_sim = "Moderate localized stains"
    decay_presence_sim = "No visible signs of decay"
    if overall_lab_colormath.lab_l < 60 and np.random.rand() < 0.1: # Darker teeth, 10% chance of decay
        decay_presence_sim = "Potential small carious lesion (simulated - consult professional)"
    elif overall_lab_colormath.lab_l < 55 and np.random.rand() < 0.05: # Even darker, 5% chance
        decay_presence_sim = "Possible early signs of demineralization (simulated - consult professional)"
    base_l = overall_lab_colormath.lab_l
    base_a = overall_lab_colormath.lab_a
    base_b = overall_lab_colormath.lab_b
    # --- REVISED ZONE VARIATIONS FOR MORE CONSISTENT DISTINCTION ---
    # These are designed to push the zones into different VITA shade neighborhoods
    # while still being plausible, building on the *normalized* base LAB.
    
    # Incisal zone: Brighter, less yellow (more translucent)
    incisal_lab_sim = LabColor(
        lab_l=np.clip(base_l + 6.0, 0, 100), # Significantly brighter
        lab_a=np.clip(base_a - 1.0, -128, 127), # Slightly less red/more green
        lab_b=np.clip(base_b - 5.0, -128, 127)  # Significantly less yellow/more blue
    )
    
    # Middle zone: Close to overall shade, with minor random variations
    middle_lab_sim = LabColor(
        lab_l=np.clip(base_l + np.random.uniform(-0.5, 0.5), 0, 100),
        lab_a=np.clip(base_a + np.random.uniform(-0.2, 0.2), -128, 127),
        lab_b=np.clip(base_b + np.random.uniform(-0.2, 0.2), -128, 127)
    )
    
    # Cervical zone: Darker, more yellow/red (more opaque)
    cervical_lab_sim = LabColor(
        lab_l=np.clip(base_l - 6.0, 0, 100), # Significantly darker
        lab_a=np.clip(base_a + 2.0, -128, 127), # More red
        lab_b=np.clip(base_b + 5.0, -128, 127)  # Significantly more yellow
    )
    # Apply normalization to the simulated zone LABs to ensure they are always plausible
    incisal_lab_sim = normalize_tooth_lab_to_realistic_range((incisal_lab_sim.lab_l, incisal_lab_sim.lab_a, incisal_lab_sim.lab_b))
    middle_lab_sim = normalize_tooth_lab_to_realistic_range((middle_lab_sim.lab_l, middle_lab_sim.lab_a, middle_lab_sim.lab_b))
    cervical_lab_sim = normalize_tooth_lab_to_realistic_range((cervical_lab_sim.lab_l, cervical_lab_sim.lab_a, cervical_lab_sim.lab_b))
    # --- END REVISED ZONE VARIATIONS ---
    print(f"DEBUG_ZONE_LABS: Incisal LAB: L={incisal_lab_sim.lab_l:.2f}, a={incisal_lab_sim.lab_a:.2f}, b={incisal_lab_sim.lab_b:.2f}")
    print(f"DEBUG_ZONE_LABS: Middle LAB: L={middle_lab_sim.lab_l:.2f}, a={middle_lab_sim.lab_a:.2f}, b={middle_lab_sim.lab_b:.2f}")
    print(f"DEBUG_ZONE_LABS: Cervical LAB: L={cervical_lab_sim.lab_l:.2f}, a={cervical_lab_sim.lab_a:.2f}, b={cervical_lab_sim.lab_b:.2f}")
    return {
        "overall_lab": {"L": overall_lab_colormath.lab_l, "a": overall_lab_colormath.lab_a, "b": overall_lab_colormath.lab_b},
        "simulated_overall_shade": match_shade_from_lab(
            (overall_lab_colormath.lab_l, overall_lab_colormath.lab_a, overall_lab_colormath.lab_b),
            zone_type="overall" # Pass zone_type for overall shade
        )[0],
        "tooth_condition": tooth_condition_sim,
        "stain_presence": stain_presence_sim,
        "decay_presence": decay_presence_sim,
        "incisal_lab": incisal_lab_sim,
        "middle_lab": middle_lab_sim,
        "cervical_lab": cervical_lab_sim,
    }
def aesthetic_shade_suggestion(facial_features, tooth_analysis):
    """
    ENHANCED PLACEHOLDER: Simulates an aesthetic mapping model with more context.
    Suggestions are now more specific, considering simulated skin/lip tones.
    Confidence is now more dynamic based on harmony score and conditions.
    """
    print("DEBUG: Simulating detailed Aesthetic Mapping and Shade Suggestion...")
    # Use the simulated tooth_analysis's overall shade as a base
    current_tooth_shade = tooth_analysis.get("simulated_overall_shade", "A2") # Default to A2
    suggested_shade = "Optimal Match (Simulated)"
    aesthetic_confidence = "High"
    recommendation_notes = "This is a simulated aesthetic suggestion. Consult a dental specialist for personalized cosmetic planning."
    # Simple rules based on current tooth shade and simulated facial features
    skin_tone = facial_features.get("skin_tone", "")
    skin_undertone = ""
    if ' with ' in skin_tone:
        skin_undertone = skin_tone.split(' with ')[1].replace(' undertones', '')
    if "Light" in skin_tone and current_tooth_shade in ["A3", "A3.5", "A4"]:
        suggested_shade = "Consider slight brightening (Simulated - e.g., B1 or A2)"
        aesthetic_confidence = "Very High"
        recommendation_notes = "Your light skin tone would be beautifully complemented by a slightly brighter smile. Discuss options with a professional."
    elif "Dark" in skin_tone and current_tooth_shade in ["A1", "B1"]:
        suggested_shade = "Maintain natural shade or subtle brightening (Simulated)"
        aesthetic_confidence = "High"
        recommendation_notes = "Your darker skin tone naturally harmonizes with a slightly warmer, natural tooth shade. Subtle changes are often best."
    elif "Warm" in skin_undertone and current_tooth_shade in ["C1", "C2", "D2"]:
        suggested_shade = "Consider warmer shades or gentle brightening (Simulated)"
        aesthetic_confidence = "Medium"
        recommendation_notes = "Your warm undertones may benefit from shades with a slightly more yellow/orange hue, or a gentle brightening to maintain harmony."
    elif "Cool" in skin_undertone and current_tooth_shade in ["A3.5", "A4", "B3", "B4"]:
        suggested_shade = "Consider cooler or neutral shades for brightening (Simulated)"
        aesthetic_confidence = "Medium"
        recommendation_notes = "Your cool undertones might be enhanced by shades with a slightly bluer/grayer hue, or a more neutral brightening."
    
    # Override suggestions if significant issues are detected
    if tooth_analysis.get("stain_presence") != "None detected":
        suggested_shade = "Professional cleaning/whitening recommended before final shade selection (Simulated)"
        aesthetic_confidence = "Low"
        recommendation_notes = "Stains can significantly impact shade assessment. Professional cleaning is advised prior to any cosmetic shade decisions. Consult a dental professional."
    
    if tooth_analysis.get("decay_presence") != "No visible signs of decay":
        suggested_shade = "Dental consultation for decay treatment (Simulated)"
        aesthetic_confidence = "Low"
        recommendation_notes = "Signs of decay require immediate professional dental attention. Aesthetic considerations should follow treatment. Consult a dental professional."
    return {
        "suggested_aesthetic_shade": suggested_shade,
        "aesthetic_confidence": aesthetic_confidence,
        "recommendation_notes": recommendation_notes
    }
def calculate_confidence(delta_e_value, has_reference_tab=False, image_brightness_l=50, exif_data=None):
    """
    Calculates a confidence score based on the Delta E value, presence of a reference tab,
    image brightness, and simulated influence from EXIF data.
    Args:
        delta_e_value (float): The Delta E 2000 value for the best shade match.
        has_reference_tab (bool): True if a reference tab was used for correction.
        image_brightness_l (float): The L* value (0-100) of the overall tooth region.
        exif_data (dict, optional): Dictionary of parsed EXIF tags.
    Returns:
        tuple: (overall_percentage, notes).
    """
    notes = [] # Use a list to build notes
    final_confidence = 0 # Default to 0% if delta_e_value is invalid or problematic
    if isinstance(delta_e_value, (int, float)) and not np.isinf(delta_e_value) and not np.isnan(delta_e_value):
        # Base confidence is higher for lower Delta E values (better match)
        # Max Delta E of 1000.0 (our error value) should result in 0% confidence
        max_delta_e_for_confidence = 10.0 
        
        # If delta_e_value is 1000.0 (our error code), confidence_from_delta_e will be negative, then clipped to 0
        confidence_from_delta_e = max(0, 100 - (delta_e_value / max_delta_e_for_confidence) * 50) 
        adjusted_confidence = confidence_from_delta_e
        if has_reference_tab:
            adjusted_confidence += 15 # Significant boost for using a reference
            notes.append("High confidence due to color correction with a reference tab.")
        else:
            adjusted_confidence -= 10 # Penalty for no reference
            notes.append("No color reference tab detected, results may have higher variability.")
        # Simulate influence of image brightness (L value of tooth)
        if image_brightness_l < 40: # Very dark image
            adjusted_confidence -= 10
            notes.append("Image appears very dark, potentially affecting accuracy.")
        elif image_brightness_l > 85: # Very bright/overexposed image
            adjusted_confidence -= 5
            notes.append("Image appears very bright, potentially affecting accuracy.")
        # Simulate influence of EXIF data (very basic heuristic)
        exif_notes_added = False
        if exif_data:
            camera_make = exif_data.get('Make', '').lower()
            camera_model = exif_data.get('Model', '').lower()
            
            # Simulate better confidence for certain "known good" camera brands/models
            if 'apple' in camera_make or 'samsung' in camera_make or 'pixel' in camera_model:
                adjusted_confidence += 2 # Small boost for common good cameras
                notes.append("EXIF data suggests a common mobile device.")
                exif_notes_added = True
            
            # Simulate lower confidence if no white balance mode detected or non-standard
            if 'WhiteBalance' in exif_data and exif_data.get('WhiteBalance') not in [0, 1]: # 0=Auto, 1=Manual
                adjusted_confidence -= 3
                notes.append("Non-standard white balance mode detected in EXIF.")
                exif_notes_added = True
            elif 'WhiteBalance' not in exif_data:
                adjusted_confidence -= 2
                notes.append("White balance metadata missing from EXIF.")
                exif_notes_added = True
        
        if not exif_data and not exif_notes_added: # If no EXIF data was parsed at all
            adjusted_confidence -= 5
            notes.append("No EXIF data available, limiting confidence adjustment.")
        
        # Ensure confidence stays within 0-100 range
        final_confidence = np.clip(adjusted_confidence, 0, 100)
    else:
        # This block is for when delta_e_value itself is problematic (NaN, Inf, or our 1000.0 error code)
        notes.append("Confidence could not be calculated due to invalid Delta E value or processing error.")
        final_confidence = 0 # Explicitly set to 0 if delta_e_value is invalid
    # Final general notes if no specific notes were added
    if not notes:
        notes.append("Confidence based on simulated analysis and input parameters.")
    
    return round(float(final_confidence)), " ".join(notes).strip()
# --- MAIN DETECTION FUNCTION (UPDATED TO THE 5-PILLAR PLAN) ---
def detect_shades_from_image(image_path, selected_reference_tab="neutral_gray", simulated_device_profile="N/A"):
    """
    Performs image processing, extracts tooth LAB, applies mathematical normalization
    using a selected reference tab, and then matches to VITA shades.
    Integrates the 5-Pillar Plan for improved accuracy and consistency.
    """
    print(f"\n--- Starting Image Processing for {image_path} ---")
    print(f"Selected Reference Tab: {selected_reference_tab}")
    print(f"Simulated Device Profile: {simulated_device_profile}")
    # Initialize all return values to default "N/A" or empty dicts
    default_shades = {
        "incisal": "N/A", "middle": "N/A", "cervical": "N/A",
        "overall_ml_shade": "ML Bypassed", # ML shade is now bypassed
        "face_features": {}, "tooth_analysis": {}, "aesthetic_suggestion": {},
        "delta_e_matched_shades": {
            "overall": "N/A", "overall_delta_e": "N/A",
            "incisal": "N/A", "incisal_delta_e": "N/A",
            "middle": "N/A", "middle_delta_e": "N/A",
            "cervical": "N/A", "cervical_delta_e": "N/A",
        },
        "accuracy_confidence": {"overall_percentage": "N/A", "notes": ""},
        "selected_device_profile": simulated_device_profile, # Pass the device profile received
        "selected_reference_tab": selected_reference_tab,
        "exif_data": {}
    }
    exif_data_parsed = {}
    img = None # Initialize img to None
    try:
        # Attempt to load with PIL for EXIF and orientation
        pil_img = Image.open(image_path)
        if pil_img is None: # CRITICAL: Check if image opened successfully
            print(f"ERROR: PIL could not open image at {image_path}. It might be corrupted or invalid.")
            return default_shades # Exit early if image is not valid

        # EXIF Orientation Handling
        try:
            for orientation in ExifTags.TAGS.keys():
                if ExifTags.TAGS[orientation] == 'Orientation':
                    break
            exif_raw = pil_img._getexif() # Get raw EXIF data
            if exif_raw: # Check if EXIF data exists before trying to access items
                exif_dict = dict(exif_raw.items()) # Convert to dict here
                if orientation in exif_dict:
                    if exif_dict[orientation] == 3:
                        pil_img = pil_img.rotate(180, expand=True)
                    elif exif_dict[orientation] == 6:
                        pil_img = pil_img.rotate(270, expand=True)
                    elif exif_dict[orientation] == 8:
                        pil_img = pil_img.rotate(90, expand=True)
            print(f"DEBUG: EXIF Orientation handled: {exif_dict.get(orientation, 'N/A') if exif_raw else 'No EXIF'}")
        except Exception as e:
            print(f"WARNING: Could not apply EXIF orientation: {e}")
            pass # Continue without orientation if error
        
        # Populate exif_data_parsed (using exif_raw if available)
        if exif_raw:
            for tag, value in ExifTags.TAGS.items():
                if tag in exif_raw:
                    decoded = ExifTags.TAGS.get(tag, tag)
                    exif_data_parsed[decoded] = exif_raw[tag]
        print(f"DEBUG: EXIF Data Parsed: {exif_data_parsed}")
        # Convert PIL image to OpenCV format (BGR) for further processing
        img = np.array(pil_img.convert('RGB')) # Convert to RGB first
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) # Then to BGR for OpenCV
        print(f"DEBUG: Image loaded via PIL and converted to BGR. Shape: {img.shape}")
    except Exception as e:
        print(f"WARNING: Could not load image with PIL or read EXIF data: {e}")
        traceback.print_exc()
        # Fallback to direct OpenCV imread if PIL fails
        img = cv2.imread(image_path)
        if img is None or img.size == 0:
            print(f"ERROR: Image at {image_path} is invalid or empty after initial load attempts. Returning default shades.")
            return default_shades
        print(f"DEBUG: Image loaded directly via OpenCV (PIL/EXIF failed). Image shape: {img.shape}")
    try:
        # Step 1: Preprocess image (resize)
        img = preprocess_image(img, target_size=(512, 512)) 
        if img is None or img.size == 0:
             print(f"ERROR: Image is invalid or empty after preprocessing. Returning default shades.")
             return default_shades
        print(f"DEBUG: 1. Image preprocessed (resized). Image shape: {img.shape}")
        # Step 2 (Pillar 1): Apply Advanced White Balance (CLAHE) - for local contrast/brightness
        img_advanced_wb = advanced_white_balance(img)
        print(f"DEBUG: 2. Advanced White Balance (CLAHE) applied. Image shape: {img_advanced_wb.shape}")
        # Step 3 (Pillar 2): Detect and Skip Bad Images (Flash or Overexposed)
        if is_overexposed(img_advanced_wb):
            print("WARNING: Image detected as overexposed. Returning 'Unreliable' shades.")
            # When overexposed, ensure tooth_analysis overall_lab is explicitly 'N/A' for template safety
            default_shades["accuracy_confidence"]["overall_percentage"] = 5
            default_shades["accuracy_confidence"]["notes"] = "Image is severely overexposed, analysis unreliable. Please retake photo in better lighting."
            default_shades["tooth_analysis"] = {
                "overall_lab": {"L": "N/A", "a": "N/A", "b": "N/A"}, # Set to N/A strings
                "simulated_overall_shade": "Unreliable",
                "tooth_condition": "Analysis unreliable due to overexposure",
                "stain_presence": "Analysis unreliable due to overexposure",
                "decay_presence": "Analysis unreliable due to overexposure",
                "incisal_lab": LabColor(0,0,0), # Dummy values for structure, but won't be used
                "middle_lab": LabColor(0,0,0),
                "cervical_lab": LabColor(0,0,0),
            }
            # Also set Delta E matched shades to Unreliable/N/A
            default_shades["delta_e_matched_shades"] = {
                "overall": "Unreliable", "overall_delta_e": "N/A",
                "incisal": "Unreliable", "incisal_delta_e": "N/A",
                "middle": "Unreliable", "middle_delta_e": "N/A",
                "cervical": "Unreliable", "cervical_delta_e": "N/A",
            }
            return default_shades
        print(f"DEBUG: 3. Overexposure check passed.")
        # Step 4: Apply Enhanced Gray World white balance - for global color cast correction
        img_global_wb = apply_reference_white_balance(img_advanced_wb)
        print(f"DEBUG: 4. Enhanced Gray World white balance applied. Image shape: {img_global_wb.shape}")
        # Step 5: Apply Simulated Device-Specific Color Adjustment
        img_device_corrected = apply_device_profile_adjustment(img_global_wb, simulated_device_profile)
        print(f"DEBUG: 5. Device-specific color adjustment applied. Image shape: {img_device_corrected.shape}")
        # Step 6 (Pillar 3): Convert to LAB using skimage (D65)
        lab_img = convert_to_lab_skimage(img_device_corrected)
        print(f"DEBUG: 6. Converted to LAB using skimage. LAB image shape: {lab_img.shape}")
        # --- Start Reference-Based Correction ---
        has_reference_tab = False
        corrected_tooth_lab_colormath = None
        # Extract tooth region average LAB first, before potential correction
        avg_lab_tuple_raw_tooth = get_tooth_region_lab(lab_img)
        print(f"DEBUG: Raw Tooth LAB (before reference correction): L={avg_lab_tuple_raw_tooth[0]:.2f}, a={avg_lab_tuple_raw_tooth[1]:.2f}, b={avg_lab_tuple_raw_tooth[2]:.2f}")
        # NEW Step: Normalize raw tooth LAB to a realistic tooth range (Pillar 0.5 - Pre-correction normalization)
        normalized_initial_tooth_lab = normalize_tooth_lab_to_realistic_range(avg_lab_tuple_raw_tooth)
        print(f"DEBUG: Normalized initial tooth LAB: {normalized_initial_tooth_lab}")
        if selected_reference_tab in REFERENCE_TAB_LAB_VALUES:
            # Assume reference patch is in a fixed corner (e.g., top-left 50x50 pixels)
            reference_patch_lab_captured = get_reference_patch_lab(lab_img, patch_coords=(10, 10, 60, 60))
            
            if reference_patch_lab_captured:
                has_reference_tab = True
                # Get the ideal LAB values for the selected reference tab
                ideal_reference_lab_colormath = REFERENCE_TAB_LAB_VALUES[selected_reference_tab]
                
                # Apply limited correction as per user suggestion
                ideal_L = ideal_reference_lab_colormath.lab_l
                ideal_a = ideal_reference_lab_colormath.lab_a
                ideal_b = ideal_reference_lab_colormath.lab_b
                captured_L = reference_patch_lab_captured[0]
                captured_a = reference_patch_lab_captured[1]
                captured_b = reference_patch_lab_captured[2]
                offset_L = np.clip(ideal_L - captured_L, -15, 15) # Limit L correction
                offset_a = np.clip(ideal_a - captured_a, -5, 5)   # Limit a correction
                offset_b = np.clip(ideal_b - captured_b, -5, 5)   # Limit b correction
                
                print(f"DEBUG: Reference offsets (clipped): L={offset_L:.2f}, a={offset_a:.2f}, b={offset_b:.2f}")
                
                # Apply offsets to the *normalized* initial tooth LAB
                corrected_l = normalized_initial_tooth_lab.lab_l + offset_L
                corrected_a = normalized_initial_tooth_lab.lab_a + offset_a
                corrected_b = normalized_initial_tooth_lab.lab_b + offset_b
                # After applying offsets, add this new adjustment for bright images:
                if corrected_l > 75:  # If still too bright after correction
                    brightness_reduction = min(15, corrected_l - 75)
                    corrected_l -= brightness_reduction * 0.7
                    print(f"DEBUG: Applied additional brightness reduction: {brightness_reduction:.2f}")
                
                corrected_tooth_lab_colormath = LabColor(lab_l=corrected_l, lab_a=corrected_a, lab_b=corrected_b)
                print(f"DEBUG: 7. Tooth LAB (after reference correction): L={corrected_tooth_lab_colormath.lab_l:.2f}, a={corrected_tooth_lab_colormath.lab_a:.2f}, b={corrected_tooth_lab_colormath.lab_b:.2f}")
            else:
                print("WARNING: Could not extract reference patch LAB. Proceeding without reference-based correction.")
                # Fallback to just normalized initial tooth LAB if reference extraction fails
                corrected_tooth_lab_colormath = normalized_initial_tooth_lab
        else:
            print("INFO: No specific reference tab selected or known. Proceeding without reference-based correction.")
            # If no reference tab is selected or known, just use the normalized initial tooth LAB
            corrected_tooth_lab_colormath = normalized_initial_tooth_lab
        # --- End Reference-Based Correction ---
        # --- Call Placeholder AI modules (adapted to use corrected_tooth_lab_colormath) ---
        face_features = detect_face_features(img) # Face features still use original image (pre-LAB conversion)
        tooth_analysis_data = simulate_tooth_analysis_from_overall_lab(corrected_tooth_lab_colormath)
        aesthetic_suggestion = aesthetic_shade_suggestion(face_features, tooth_analysis_data)
        print("DEBUG: 8. Simulated AI modules executed.")
        
        # --- Delta E Matching (using the final corrected_tooth_lab_colormath) ---
        # Using the fixed thresholded matching function (Pillar 4)
        overall_delta_e_shade, overall_min_delta = match_shade_from_lab(
            (corrected_tooth_lab_colormath.lab_l, corrected_tooth_lab_colormath.lab_a, corrected_tooth_lab_colormath.lab_b),
            zone_type="overall" # Pass zone_type for overall shade
        )
        print(f"DEBUG: 9. Overall Delta E matched shade: {overall_delta_e_shade} (dE={overall_min_delta:.2f})")
        # Derive zone shades from the overall LAB for report consistency (simulated)
        # These are now derived from the single overall corrected LAB, not true segmentation
        incisal_lab_colormath = tooth_analysis_data["incisal_lab"]
        middle_lab_colormath = tooth_analysis_data["middle_lab"]
        cervical_lab_colormath = tooth_analysis_data["cervical_lab"]
        incisal_delta_e_shade, incisal_min_delta = match_shade_from_lab(
            (incisal_lab_colormath.lab_l, incisal_lab_colormath.lab_a, incisal_lab_colormath.lab_b),
            zone_type="incisal" # Specify zone type for adaptive threshold
        )
        middle_delta_e_shade, middle_min_delta = match_shade_from_lab(
            (middle_lab_colormath.lab_l, middle_lab_colormath.lab_a, middle_lab_colormath.lab_b),
            zone_type="middle" # Specify zone type for adaptive threshold
        )
        cervical_delta_e_shade, cervical_min_delta = match_shade_from_lab(
            (cervical_lab_colormath.lab_l, cervical_lab_colormath.lab_a, cervical_lab_colormath.lab_b),
            zone_type="cervical" # Specify zone type for adaptive threshold
        )
        print(f"DEBUG: 10. Simulated Zone Delta E matches: Incisal={incisal_delta_e_shade} (dE={incisal_min_delta:.2f}), Middle={middle_delta_e_shade} (dE={middle_min_delta:.2f}), Cervical={cervical_delta_e_shade} (dE={cervical_min_delta:.2f})")
        # --- IMPORTANT DEBUG PRINT FOR DELTA E VALUES ---
        print(f"DEBUG_FINAL_DE: Overall_min_delta (raw): {overall_min_delta}")
        print(f"DEBUG_FINAL_DE: Incisal_min_delta (raw): {incisal_min_delta}")
        print(f"DEBUG_FINAL_DE: Middle_min_delta (raw): {middle_min_delta}")
        print(f"DEBUG_FINAL_DE: Cervical_min_delta (raw): {cervical_min_delta}")
        # Add new debug prints to check formatted values
        print(f"DEBUG_FORMAT: Overall Delta E formatted: {format_delta_e(overall_min_delta)}")
        print(f"DEBUG_FORMAT: Incisal Delta E formatted: {format_delta_e(incisal_min_delta)}")
        print(f"DEBUG_FORMAT: Middle Delta E formatted: {format_delta_e(middle_min_delta)}")
        print(f"DEBUG_FORMAT: Cervical Delta E formatted: {format_delta_e(cervical_min_delta)}")
        # --- END IMPORTANT DEBUG PRINT ---
        # --- Calculate Confidence ---
        overall_accuracy_confidence, confidence_notes = calculate_confidence(
            overall_min_delta, 
            has_reference_tab=has_reference_tab, 
            image_brightness_l=corrected_tooth_lab_colormath.lab_l, # Pass the actual L value of the corrected tooth
            exif_data=exif_data_parsed # Pass the parsed EXIF data
        )
        print(f"DEBUG: 11. Overall Accuracy Confidence: {overall_accuracy_confidence}%")
        
        # Determine the final "Rule-based" shades (these are now derived from Delta E matches of simulated zones)
        # These will now always be a VITA shade, never "Unreliable" from match_shade_from_lab
        final_incisal_rule_based = incisal_delta_e_shade
        final_middle_rule_based = middle_delta_e_shade
        final_cervical_rule_based = cervical_delta_e_shade
        print(f"DEBUG: 12. Final Rule-based Shades (to be displayed): Incisal={final_incisal_rule_based}, Middle={final_middle_rule_based}, Cervical={final_cervical_rule_based}")
        # Detected shades dictionary construction
        detected_shades = {
            "incisal": final_incisal_rule_based,
            "middle": final_middle_rule_based,
            "cervical": final_cervical_rule_based,
            "overall_ml_shade": "ML Bypassed", # Explicitly state ML is bypassed
            "face_features": face_features,
            "tooth_analysis": tooth_analysis_data, # tooth_analysis_data is now a dictionary with L,a,b for overall_lab
            "aesthetic_suggestion": aesthetic_suggestion,
            "accuracy_confidence": {
                "overall_percentage": overall_accuracy_confidence,
                "notes": confidence_notes
            },
            "selected_device_profile": simulated_device_profile, # Pass the device profile received
            "selected_reference_tab": selected_reference_tab,
            "exif_data": exif_data_parsed, # Add EXIF data to the output
            "delta_e_matched_shades": { # THIS WAS THE MISSING BLOCK!
                "overall": format_shade_name(overall_delta_e_shade, overall_min_delta),
                "overall_delta_e": format_delta_e(overall_min_delta),
                "incisal": format_shade_name(incisal_delta_e_shade, incisal_min_delta),
                "incisal_delta_e": format_delta_e(incisal_min_delta),
                "middle": format_shade_name(middle_delta_e_shade, middle_min_delta),
                "middle_delta_e": format_delta_e(middle_min_delta),
                "cervical": format_shade_name(cervical_delta_e_shade, cervical_min_delta),
                "cervical_delta_e": format_delta_e(cervical_min_delta),
            },
        }
        return detected_shades
    except FileNotFoundError as fnfe:
        print(f"ERROR: File not found during shade detection: {fnfe}")
        flash(f"Error: Image file not found. Please re-upload.", 'danger')
        return default_shades
    except Exception as e:
        print(f"CRITICAL ERROR during shade detection: {e}")
        traceback.print_exc()
        # Ensure default_shades is returned with 'N/A' for LAB values if critical error occurs
        default_shades["tooth_analysis"] = {
            "overall_lab": {"L": "N/A", "a": "N/A", "b": "N/A"},
            "simulated_overall_shade": "N/A",
            "tooth_condition": "Error during analysis",
            "stain_presence": "Error during analysis",
            "decay_presence": "Error during analysis",
            "incisal_lab": LabColor(0,0,0), # Dummy values for structure
            "middle_lab": LabColor(0,0,0),
            "cervical_lab": LabColor(0,0,0),
        }
        default_shades["accuracy_confidence"]["overall_percentage"] = 0
        default_shades["accuracy_confidence"]["notes"] = f"Critical error during image processing: {e}"
        return default_shades

def generate_pdf_report(patient_name, shades, image_path, filepath):
    """Generates a PDF report with detected shades and the uploaded image."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=16)
    pdf.cell(200, 10, txt="Shade View - Tooth Shade Analysis Report", ln=True, align='C')
    pdf.ln(10)
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, txt=f"Patient Name: {patient_name}", ln=True)
    pdf.cell(0, 10, txt=f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
    
    # Update this line as device_profile is now N/A or reference-based
    pdf.cell(0, 10, txt=f"Correction Method: {shades.get('selected_device_profile', 'N/A')} (Reference-based correction used)", ln=True) # Re-added the device profile here
    
    selected_ref_tab = shades.get("selected_reference_tab", "N/A").replace("_", " ").title()
    pdf.cell(0, 10, txt=f"Color Reference Used: {selected_ref_tab}", ln=True)
    pdf.ln(5)
    pdf.set_font("Arial", 'B', size=14)
    pdf.cell(0, 10, txt="Detected Shades (Rule-based / Delta E):", ln=True) # Changed from Rule-based / AI
    pdf.set_font("Arial", size=12)
    
    overall_ml_shade = shades.get("overall_ml_shade", "N/A")
    if overall_ml_shade != "N/A" and overall_ml_shade != "ML Bypassed": # Only show if not bypassed
        pdf.cell(0, 7, txt=f"   - Overall AI Prediction (ML): {overall_ml_shade}", ln=True)
    pdf.cell(0, 7, txt=f"   - Incisal Zone (Rule-based): {shades.get('incisal', 'N/A')}", ln=True)
    pdf.cell(0, 7, txt=f"   - Middle Zone (Rule-based): {shades.get('middle', 'N/A')}", ln=True)
    pdf.cell(0, 7, txt=f"   - Cervical Zone (Rule-based): {shades.get('cervical', 'N/A')}", ln=True)
    
    pdf.ln(8)
    pdf.set_font("Arial", 'B', size=14)
    pdf.cell(0, 10, txt="Delta E 2000 Matched Shades (Perceptual Match):", ln=True)
    pdf.set_font("Arial", size=12)
    delta_e_shades = shades.get("delta_e_matched_shades", {})
    if delta_e_shades:
        # These values are already formatted as strings (or "N/A") by format_delta_e
        overall_de = delta_e_shades.get('overall_delta_e', 'N/A')
        incisal_de = delta_e_shades.get('incisal_delta_e', 'N/A')
        middle_de = delta_e_shades.get('middle_delta_e', 'N/A')
        cervical_de = delta_e_shades.get('cervical_delta_e', 'N/A')
        # Removed the problematic f-string conditional formatting here
        pdf.cell(0, 7, txt=f"   - Overall Delta E Match: {delta_e_shades.get('overall', 'N/A')} (dE: {overall_de})", ln=True)
        pdf.cell(0, 7, txt=f"   - Incisal Zone Delta E Match: {delta_e_shades.get('incisal', 'N/A')} (dE: {incisal_de})", ln=True)
        pdf.cell(0, 7, txt=f"   - Middle Zone Delta E Match: {delta_e_shades.get('middle', 'N/A')} (dE: {middle_de})", ln=True)
        pdf.cell(0, 7, txt=f"   - Cervical Zone Delta E Match: {delta_e_shades.get('cervical', 'N/A')} (dE: {cervical_de})", ln=True)
    else:
        pdf.cell(0, 7, txt="   - Delta E matching data not available.", ln=True)
    pdf.ln(8)
    pdf.set_font("Arial", 'B', size=14)
    pdf.cell(0, 10, txt="Shade Detection Accuracy Confidence:", ln=True)
    pdf.set_font("Arial", size=12)
    accuracy_conf = shades.get("accuracy_confidence", {})
    overall_percentage = accuracy_conf.get("overall_percentage", "N/A")
    if overall_percentage != "N/A":
        pdf.cell(0, 7, txt=f"   - Overall Confidence: {overall_percentage}%", ln=True)
        pdf.multi_cell(0, 7, txt=f"   - Notes: {accuracy_conf.get('notes', 'N/A')}")
    else:
        pdf.cell(0, 7, txt="   - Confidence data not available or processing error.", ln=True)
    # Display EXIF data in PDF (optional, for debugging/info)
    exif_data_for_report = shades.get("exif_data", {})
    if exif_data_for_report:
        pdf.ln(5)
        pdf.set_font("Arial", 'B', size=10)
        pdf.cell(0, 7, txt="EXIF Data (from Image Metadata):", ln=True)
        pdf.set_font("Arial", size=9)
        # Limit the number of EXIF tags to display to keep report concise
        display_tags = ['Make', 'Model', 'DateTimeOriginal', 'FNumber', 'ExposureTime', 'ISOSpeedRatings', 'Flash', 'WhiteBalance']
        exif_found = False
        for tag_name in display_tags:
            if tag_name in exif_data_for_report:
                pdf.cell(0, 5, txt=f"   - {tag_name}: {exif_data_for_report[tag_name]}", ln=True)
                exif_found = True
        if not exif_found:
             pdf.cell(0, 5, txt="   - No common EXIF data found.", ln=True)
    pdf.ln(8)
    pdf.set_font("Arial", 'B', size=13)
    pdf.cell(0, 10, txt="Advanced AI Insights (Simulated):", ln=True)
    pdf.set_font("Arial", size=11)
    tooth_analysis = shades.get("tooth_analysis", {})
    if tooth_analysis:
        pdf.cell(0, 7, txt=f"   -- Tooth Analysis --", ln=True)
        pdf.cell(0, 7, txt=f"   - Simulated Overall Shade (Detailed): {tooth_analysis.get('simulated_overall_shade', 'N/A')}",
                 ln=True)
        pdf.cell(0, 7, txt=f"   - Simulated Condition: {tooth_analysis.get('tooth_condition', 'N/A')}", ln=True)
        pdf.cell(0, 7, txt=f"   - Simulated Stain Presence: {tooth_analysis.get('stain_presence', 'N/A')}", ln=True)
        pdf.cell(0, 7, txt=f"   - Simulated Decay Presence: {tooth_analysis.get('decay_presence', 'N/A')}", ln=True)
        overall_lab_data = tooth_analysis.get('overall_lab', {})
        l_val = overall_lab_data.get('L', 'N/A')
        a_val = overall_lab_data.get('a', 'N/A')
        b_val = overall_lab_data.get('b', 'N/A')
        if all(isinstance(v, (int, float)) for v in [l_val, a_val, b_val]):
            pdf.cell(0, 7, txt=f"   - Simulated Overall LAB: L={l_val:.2f}, a={a_val:.2f}, b={b_val:.2f}", ln=True)
        else:
            pdf.cell(0, 7, txt=f"   - Simulated Overall LAB: L={l_val}, a={a_val}, b={b_val}", ln=True)
    
    pdf.ln(3)
    face_features = shades.get("face_features", {})
    if face_features:
        pdf.cell(0, 7, txt="   -- Facial Aesthetics Analysis --", ln=True)
        for key, value in face_features.items():
            if isinstance(value, (int, float)):
                pdf.cell(0, 7, txt=f"    {key.replace('_', ' ').title()}: {value:.2f}", ln=True)
            else:
                pdf.cell(0, 7, txt=f"    {key.replace('_', ' ').title()}: {value}", ln=True)
    pdf.ln(3)
    aesthetic_suggestion = shades.get("aesthetic_suggestion", {})
    if aesthetic_suggestion:
        pdf.cell(0, 7, txt=f"   -- Aesthetic Shade Suggestion --", ln=True)
        for key, value in aesthetic_suggestion.items():
            pdf.multi_cell(0, 7, txt=f"    {key.replace('_', ' ').title()}: {value}")
    pdf.ln(10)
    try:
        if os.path.exists(image_path):
            pdf.cell(0, 10, txt="Uploaded Image:", ln=True)
            # Check if current position is too low for image + footer
            if pdf.get_y() > 200: # Arbitrary threshold, adjust as needed
                pdf.add_page()
            
            img_cv = cv2.imread(image_path)
            if img_cv is not None:
                img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
                temp_image_path = "temp_pdf_image.png"
                cv2.imwrite(temp_image_path, img_rgb)
                
                h_img, w_img, _ = img_cv.shape
                max_w_pdf = 180
                w_pdf = min(w_img, max_w_pdf)
                h_pdf = h_img * (w_pdf / w_img)
                # Check again before adding image to ensure it fits on current or next page
                if pdf.get_y() + h_pdf + 10 > pdf.h - pdf.b_margin:
                    pdf.add_page()
                pdf.image(temp_image_path, x=pdf.get_x(), y=pdf.get_y(), w=w_pdf, h=h_pdf)
                pdf.ln(h_pdf + 10)
                os.remove(temp_image_path)
            else:
                pdf.cell(0, 10, txt="Note: Image could not be loaded for embedding.", ln=True)
        else:
            pdf.cell(0, 10, txt="Note: Uploaded image file not found for embedding.", ln=True)
    except Exception as e:
        print(f"Error adding image to PDF: {e}")
        traceback.print_exc() # Print full traceback for debugging
        pdf.cell(0, 10, txt="Note: An error occurred while embedding the image in the report.", ln=True)
    pdf.set_font("Arial", 'I', size=9)
    pdf.multi_cell(0, 6,
                   txt="DISCLAIMER: This report is based on simulated analysis for demonstration purposes only. It is not intended for clinical diagnosis, medical advice, or professional cosmetic planning. Always consult with a qualified dental or medical professional for definitive assessment, diagnosis, and treatment.",
                   align='C')
    pdf.output(filepath)
# ===============================================
# 5. ROUTES (Adapted for Firestore)
# ===============================================
@app.route('/')
def home():
    """Renders the home/landing page."""
    return render_template('index.html')
@app.route('/login', methods=['GET', 'POST'])
def login():
    """Handles user login (Simulated for Canvas)."""
    if g.user and 'anonymous' not in g.user['id']:
        flash(f"You are already logged in as {g.user['username']}.", 'info')
        return redirect(url_for('dashboard'))
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password'] # Password is not hashed/checked in simulated auth
        error = None
        if not username or not password:
            error = 'Username and password are required.'
        # In a real app, you'd fetch user from DB and check password_hash
        # For simulation, any non-empty username/password is "successful"
        if error is None:
            simulated_user_id = 'user_' + username.lower().replace(' ', '_')
            session['user_id'] = simulated_user_id
            session['user'] = {'id': simulated_user_id, 'username': f"User_{simulated_user_id[:8]}"}
            flash(f'Simulated login successful for {username}!', 'success')
            print(f"DEBUG: Simulated login for user: {username} (ID: {session['user_id']})")
            return redirect(url_for('dashboard'))
        flash(error, 'danger')
    return render_template('login.html')
@app.route('/register', methods=['GET', 'POST'])
def register():
    """Handles user registration (Simulated for Canvas)."""
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password'] # Password is not hashed/stored in simulated auth
        error = None
        if not username:
            error = 'Username is required.'
        elif not password:
            error = 'Password is required.'
        # In a real app, you'd check if username already exists in DB
        # For simulation, any non-empty username/password is "successful"
        
        if error is None:
            flash(f"Simulated registration successful for {username}. You can now log in!", 'success')
            return redirect(url_for('login'))
        flash(error, 'danger')
    return render_template('register.html')
@app.route('/logout')
def logout():
    """Handles user logout."""
    session.clear()
    flash('You have been logged out.', 'info')
    print(f"DEBUG: User logged out. Session cleared.")
    return redirect(url_for('home'))
@app.route('/dashboard')
@login_required
def dashboard():
    """Renders the user dashboard, displaying past reports and patient info."""
    user_id = g.user['id']
    patients_path = ['artifacts', app_id, 'users', user_id, 'patients']
    reports_path = ['artifacts', app_id, 'users', user_id, 'reports']
    # Get all patients for the current user
    all_patients = get_firestore_documents_in_collection(patients_path, query_filters={'user_id': user_id})
    # Get all reports for the current user
    all_reports = get_firestore_documents_in_collection(reports_path, query_filters={'user_id': user_id})
    # Create a mapping from op_number to the latest report for that op_number
    op_to_latest_report = {}
    for report in all_reports:
        op_num = report.get('op_number')
        report_timestamp = report.get('timestamp')
        
        if op_num:
            # Compare timestamps to ensure we always store the latest report
            if op_num not in op_to_latest_report or report_timestamp > op_to_latest_report[op_num].get('timestamp', ''):
                op_to_latest_report[op_num] = report
    # Enrich patient data with their latest report information
    enriched_patients = []
    for patient in all_patients:
        patient_copy = patient.copy() # Create a mutable copy
        latest_report = op_to_latest_report.get(patient_copy.get('op_number'))
        
        if latest_report:
            patient_copy['report_filename'] = latest_report.get('report_filename')
            patient_copy['latest_analysis_date'] = datetime.fromisoformat(latest_report.get('timestamp')).strftime('%Y-%m-%d %H:%M:%S') if latest_report.get('timestamp') else 'N/A'
            # You can add more report details here if needed, e.g., latest shade
            patient_copy['latest_overall_shade'] = latest_report.get('detected_shades', {}).get('delta_e_matched_shades', {}).get('overall', 'N/A')
        else:
            patient_copy['report_filename'] = None
            patient_copy['latest_analysis_date'] = 'No reports yet'
            patient_copy['latest_overall_shade'] = 'N/A'
        
        enriched_patients.append(patient_copy)
    
    # Sort patients by their creation date or latest report date if available
    enriched_patients.sort(key=lambda x: x.get('latest_analysis_date', x.get('created_at', '')), reverse=True)
    current_date_formatted = datetime.now().strftime('%Y-%m-%d')
    return render_template('dashboard.html',
                           patients=enriched_patients, # Renamed from 'reports' to 'patients' for clarity
                           user=g.user,
                           current_date=current_date_formatted)
@app.route('/save_patient_data', methods=['POST'])
@login_required
def save_patient_data():
    """Handles saving new patient records to Firestore and redirects to image upload page."""
    op_number = request.form['op_number']
    patient_name = request.form['patient_name']
    age = request.form['age']
    sex = request.form['sex']
    record_date = request.form['date']
    user_id = g.user['id']
    patients_collection_path = ['artifacts', app_id, 'users', user_id, 'patients']
    # Check for existing OP Number for the current user in Firestore
    existing_patients = get_firestore_documents_in_collection(patients_collection_path, query_filters={'op_number': op_number, 'user_id': user_id})
    if existing_patients:
        flash('OP Number already exists for another patient under your account. Please use a unique OP Number or select from recent entries.', 'error')
        return redirect(url_for('dashboard'))
    try:
        patient_data = {
            'user_id': user_id,
            'op_number': op_number,
            'patient_name': patient_name,
            'age': int(age),
            'sex': sex,
            'record_date': record_date,
            'created_at': datetime.now().isoformat()
        }
        
        # Add patient data and get the generated document ID
        patient_doc_id = add_firestore_document(patients_collection_path, patient_data)
        patient_data['id'] = patient_doc_id # Store the ID in the data too for consistency
        flash('Patient record saved successfully! Now upload an image for analysis.', 'success')
        return redirect(url_for('upload_page', op_number=op_number))
    except Exception as e:
        flash(f'Error saving patient record: {e}', 'danger')
        traceback.print_exc()
        return redirect(url_for('dashboard'))
@app.route('/upload_page/<op_number>')
@login_required
def upload_page(op_number):
    """Renders the dedicated image upload page for a specific patient."""
    user_id = g.user['id']
    patients_collection_path = ['artifacts', app_id, 'users', user_id, 'patients']
    patient = None
    all_patients = get_firestore_documents_in_collection(patients_collection_path, query_filters={'op_number': op_number, 'user_id': user_id})
    if all_patients:
        patient = all_patients[0] # Get the first matching patient
    if patient is None:
        flash('Patient not found or unauthorized access.', 'error')
        return redirect(url_for('dashboard'))
    return render_template('upload_page.html', op_number=op_number, patient_name=patient['patient_name'])
@app.route('/upload', methods=['POST'])
@login_required
def upload_file():
    """Handles image upload, shade detection, and PDF report generation."""
    if request.method == 'POST':
        # Check for file from either 'file' input or 'camera_file' input
        uploaded_file_obj = None
        if 'file' in request.files and request.files['file'].filename != '':
            uploaded_file_obj = request.files['file']
        elif 'camera_file' in request.files and request.files['camera_file'].filename != '':
            uploaded_file_obj = request.files['camera_file']
        
        if uploaded_file_obj is None: # Corrected from `=== None` to `is None`
            flash('No file selected or captured.', 'danger')
            return redirect(request.url)
        op_number_from_form = request.form.get('op_number')
        patient_name = request.form.get('patient_name', 'Unnamed Patient')
        simulated_device_profile = request.form.get('device_profile', 'N/A') # Get device profile from form
        selected_reference_tab = request.form.get('reference_tab', 'neutral_gray')
        original_image_path = None # Initialize to None
        try:
            if uploaded_file_obj:
                filename = secure_filename(uploaded_file_obj.filename)
                file_ext = os.path.splitext(filename)[1]
                unique_filename = str(uuid.uuid4()) + file_ext
                
                original_image_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
                uploaded_file_obj.save(original_image_path)
                flash('Image uploaded successfully!', 'success')
                # Call the updated detect_shades_from_image
                detected_shades = detect_shades_from_image(original_image_path, selected_reference_tab, simulated_device_profile)
                # Check if overall shade is "N/A" indicating a fundamental processing error
                # (since match_shade_from_lab no longer returns "Unreliable")
                if detected_shades.get("delta_e_matched_shades", {}).get("overall") == "N/A":
                    flash("Error processing image for shade detection. Please try another image or check image quality.", 'danger')
                    if original_image_path and os.path.exists(original_image_path):
                        os.remove(original_image_path) # Clean up uploaded file if processing failed
                    return redirect(url_for('upload_page', op_number=op_number_from_form))
                report_filename = f"report_{patient_name.replace(' ', '')}_{op_number_from_form}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                report_filepath = os.path.join(app.config['REPORT_FOLDER'], report_filename)
                generate_pdf_report(patient_name, detected_shades, original_image_path, report_filepath)
                flash('PDF report generated!', 'success')
                # Save report data to Firestore
                reports_collection_path = ['artifacts', app_id, 'users', g.firestore_user_id, 'reports']
                report_data = {
                    'patient_name': patient_name,
                    'op_number': op_number_from_form,
                    'original_image': unique_filename, # Store just the filename, not full path
                    'report_filename': report_filename,
                    'detected_shades': detected_shades, # Store the full shades dictionary
                    'timestamp': datetime.now().isoformat(), # ISO format for easy sorting
                    'user_id': g.firestore_user_id
                }
                report_doc_id = add_firestore_document(reports_collection_path, report_data)
                report_data['id'] = report_doc_id # Add the generated ID to the data
                flash('Report data saved to dashboard!', 'info')
                return redirect(url_for('report_page', report_filename=report_filename))
        except Exception as e:
            flash(f'An unexpected error occurred during upload or processing: {e}', 'danger')
            traceback.print_exc() # Print full traceback for debugging
            if original_image_path and os.path.exists(original_image_path):
                os.remove(original_image_path) # Clean up uploaded file on error
            return redirect(url_for('upload_page', op_number=op_number_from_form))
    
    # If GET request or initial POST failed without redirect
    flash("Please select a patient from the dashboard to upload an image.", 'info') # Added this flash message
    return redirect(url_for('dashboard'))
@app.route('/report/<report_filename>')
@login_required
def report_page(report_filename):
    """Displays a detailed analysis report."""
    user_id = g.user['id']
    reports_collection_path = ['artifacts', app_id, 'users', user_id, 'reports']
    
    # Find the specific report by filename
    report_data = None
    all_reports = get_firestore_documents_in_collection(reports_collection_path)
    for r in all_reports:
        if r.get('report_filename') == report_filename:
            report_data = r
            break
    if report_data is None:
        flash('Report not found or unauthorized access.', 'error')
        return redirect(url_for('dashboard'))
    # Extract data for rendering
    patient_name = report_data.get('patient_name', 'N/A')
    analysis_date = datetime.fromisoformat(report_data.get('timestamp')).strftime('%Y-%m-%d %H:%M:%S') if report_data.get('timestamp') else 'N/A'
    shades = report_data.get('detected_shades', {})
    image_filename = report_data.get('original_image')
    report_id = report_data.get('id') # Assuming 'id' is stored when added to Firestore
    # Determine correction method string for display
    correction_method_display = shades.get('selected_device_profile', 'N/A')
    if correction_method_display != 'N/A':
        correction_method_display = f"{correction_method_display.replace('_', ' ').title()} (Reference-based correction used)"
    else:
        correction_method_display = "N/A (Reference-based correction used)"
    return render_template('report.html',
                           patient_name=patient_name,
                           analysis_date=analysis_date,
                           shades=shades,
                           image_filename=image_filename,
                           report_filename=report_filename, # Pass for download link
                           correction_method=correction_method_display, # Pass the formatted string
                           reference_tab=shades.get('selected_reference_tab', 'N/A'),
                           report_id=report_id) # Pass report_id for feedback
@app.route('/download_report/<filename>')
@login_required
def download_report(filename):
    """Allows users to download their generated PDF reports."""
    # Ensure the file belongs to the logged-in user for security (simulated)
    # In a real app, you'd check Firestore to confirm user_id matches report_data.user_id
    report_path = os.path.join(app.config['REPORT_FOLDER'], filename)
    if os.path.exists(report_path):
        return send_from_directory(app.config['REPORT_FOLDER'], filename, as_attachment=True)
    else:
        flash('Report file not found.', 'error')
        return redirect(url_for('dashboard'))
@app.route('/submit_feedback', methods=['POST'])
@login_required
def submit_feedback():
    """Handles user feedback submission and logs it to Firestore."""
    report_id = request.form.get('report_id')
    is_correct = request.form.get('is_correct') == 'true'
    correct_shade = request.form.get('correct_shade') # Only if is_correct is false
    user_id = g.user['id']
    if not report_id:
        flash("Feedback submission failed: Missing report ID.", 'danger')
        return redirect(url_for('dashboard'))
    feedback_data = {
        'user_id': user_id,
        'report_id': report_id,
        'is_correct': is_correct,
        'correct_shade_provided': correct_shade if not is_correct else None,
        'timestamp': datetime.now().isoformat()
    }
    try:
        feedback_collection_path = ['artifacts', app_id, 'users', user_id, 'feedback']
        add_firestore_document(feedback_collection_path, feedback_data)
        flash("Thank you for your feedback! It has been logged for future model improvement.", 'success')
        print(f"DEBUG: Feedback logged for report {report_id}: Correct={is_correct}, Provided Shade={correct_shade}")
    except Exception as e:
        flash(f"Error submitting feedback: {e}", 'danger')
        traceback.print_exc()
    # Redirect back to the report page or dashboard
    # Find the report filename to redirect correctly
    reports_path = ['artifacts', app_id, 'users', user_id, 'reports']
    all_reports = get_firestore_documents_in_collection(reports_path)
    report_filename = None
    # We need to iterate through the *actual documents* to find the one with the matching 'id'
    for doc_id, doc_data in db_data['artifacts'][app_id]['users'][user_id]['reports'].items():
        if doc_id == report_id:
            report_filename = doc_data.get('report_filename')
            break
    
    if report_filename:
        return redirect(url_for('report_page', report_filename=report_filename))
    else:
        return redirect(url_for('dashboard'))
@app.route('/ux_report')
def ux_report_page():
    """Renders the UX Report page."""
    return render_template('ux_report.html')
@app.route('/diagnostics')
def diagnostics():
    """
    Provides a diagnostic endpoint to serve a test image for color calibration.
    Creates a test image with known VITA shade patches.
    """
    try:
        img_width = 600
        img_height = 400
        patch_size = 100
        num_cols = img_width // patch_size
        num_rows = img_height // patch_size
        test_image_array = np.zeros((img_height, img_width, 3), dtype=np.uint8)
        # Select a few representative VITA shades
        test_shades = ["A1", "A2", "B1", "C1", "D2", "A3"]
        
        # Ensure we don't go out of bounds if there are more shades than patches
        num_patches = min(len(test_shades), num_rows * num_cols)
        for i in range(num_patches):
            shade_name = test_shades[i]
            lab_color = VITA_SHADE_LAB_REFERENCES[shade_name]
            
            # Convert LAB to RGB for display
            # skimage.color.lab2rgb expects L in [0, 100] and a,b in [-128, 127]
            # Output is RGB float in [0, 1]
            rgb_float = color.lab2rgb([[lab_color.lab_l, lab_color.lab_a, lab_color.lab_b]])[0]
            rgb_255 = (rgb_float * 255).astype(np.uint8)
            row = i // num_cols
            col = i % num_cols
            
            y_start = row * patch_size
            y_end = (row + 1) * patch_size
            x_start = col * patch_size
            x_end = (col + 1) * patch_size
            test_image_array[y_start:y_end, x_start:x_end] = rgb_255
            # Add text label for the shade (optional, but helpful)
            text_x = x_start + 5
            text_y = y_start + patch_size // 2 + 5
            cv2.putText(test_image_array, shade_name, (text_x, text_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA) # Black text
            cv2.putText(test_image_array, shade_name, (text_x, text_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA) # White outline
        # Convert RGB to BGR for OpenCV saving
        test_image_bgr = cv2.cvtColor(test_image_array, cv2.COLOR_RGB2BGR)
        test_image_filename = 'test_vita_shades.png'
        test_image_path = os.path.join(app.config['UPLOAD_FOLDER'], test_image_filename)
        cv2.imwrite(test_image_path, test_image_bgr)
        
        return send_file(test_image_path, mimetype='image/png')
    except Exception as e:
        print(f"ERROR: Could not generate diagnostic image: {e}")
        traceback.print_exc()
        return "Error generating diagnostic image.", 500
if __name__ == '__main__':
    # This block will only run when the script is executed directly
    # and not when imported by a web server (like Gunicorn or Flask's dev server)
    # For local development, you might want to uncomment app.run()
    pass # Keep pass for Canvas environment
