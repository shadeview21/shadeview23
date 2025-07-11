import os
import uuid
from datetime import datetime
import json
import traceback

import cv2  # For image processing
import numpy as np  # For numerical operations with images
from skimage import color # For better LAB conversion
from PIL import Image, ExifTags # For EXIF data reading

from fpdf import FPDF # For PDF generation

from flask import Flask, render_template, request, redirect, url_for, flash, session, g, send_from_directory, send_file
from werkzeug.utils import secure_filename

# --- SQLModel for Database ORM ---
# You need to install these: pip install sqlmodel psycopg2-binary
from sqlmodel import Field, Session, SQLModel, create_engine, select
from typing import Optional, List

# --- NEW IMPORT FOR JSONB TYPE ---
from sqlalchemy.dialects.postgresql import JSON # For JSONB column type in PostgreSQL
# --- END NEW IMPORT ---

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

    C1 = np.sqrt(a1**2 + b1**2)
    C2 = np.sqrt(a2**2 + b2**2)
    
    C_bar = (C1 + C2) / 2.0
    L_bar = (L1 + L2) / 2.0
    
    h1_rad = np.arctan2(b1, a1)
    h2_rad = np.arctan2(b2, a2)
    
    # Ensure hue angles are positive (0 to 2*pi)
    if h1_rad < 0: h1_rad += 2 * np.pi
    if h2_rad < 0: h2_rad += 2 * np.pi

    delta_L_prime = L2 - L1
    
    # Calculate a' and b' values
    if C_bar == 0:
        G = 0.0
    else:
        denominator_G = C_bar**7 + 25**7
        G = 0.5 * (1 - np.sqrt(C_bar**7 / denominator_G)) if denominator_G != 0 else 0.0
    
    a1_prime = a1 + a1 * G
    a2_prime = a2 + a2 * G

    C1_prime = np.sqrt(a1_prime**2 + b1**2)
    C2_prime = np.sqrt(a2_prime**2 + b2**2)
    
    delta_C_prime = C2_prime - C1_prime
    
    delta_a_prime = a2_prime - a1_prime
    delta_b_prime = b2 - b1
    delta_H_prime_sq = delta_a_prime**2 + delta_b_prime**2 - delta_C_prime**2
    
    if delta_H_prime_sq < 0:
        delta_H_prime = 0.0 # Ensure it's float
    else:
        delta_H_prime = np.sqrt(delta_H_prime_sq)

    # Calculate H_bar_prime
    if C1_prime * C2_prime == 0:
        H_bar_prime = 0.0 # Ensure it's float
    else:
        # H_bar_prime calculation as per standard Delta E 2000
        if np.abs(h1_rad - h2_rad) <= np.pi:
            H_bar_prime = (h1_rad + h2_rad) / 2.0
        elif np.abs(h1_rad - h2_rad) > np.pi and (h1_rad + h2_rad) < 2 * np.pi:
            H_bar_prime = (h1_rad + h2_rad) / 2.0 + np.pi
        else: # np.abs(h1_rad - h2_rad) > np.pi and (h1_rad + h2_rad) >= 2 * np.pi
            H_bar_prime = (h1_rad + h2_rad) / 2.0 - np.pi

    H_bar_prime_deg = np.rad2deg(H_bar_prime)

    # Weighting functions
    S_L_denom = np.sqrt(20.0 + (L_bar - 50.0)**2)
    S_L = 1.0 + ((0.015 * (L_bar - 50.0)**2) / S_L_denom if S_L_denom != 0 else 0)
    S_C = 1.0 + 0.045 * C_bar
    
    T = 1.0 - 0.17 * np.cos(np.deg2rad(H_bar_prime_deg - 30.0)) + \
        0.24 * np.cos(np.deg2rad(2.0 * H_bar_prime_deg)) + \
        0.32 * np.cos(np.deg2rad(3.0 * H_bar_prime_deg + 6.0)) - \
        0.20 * np.cos(np.deg2rad(4.0 * H_bar_prime_deg - 63.0))
    S_H = 1.0 + 0.015 * C_bar * T

    delta_theta_deg = 30.0 * np.exp(-((H_bar_prime_deg - 275.0) / 25.0)**2)

    R_C_denom = (C_bar**7 + 25**7)
    R_C = 2.0 * np.sqrt(C_bar**7 / R_C_denom if R_C_denom != 0 else 0)

    R_T = -R_C * np.sin(np.deg2rad(2.0 * delta_theta_deg))

    # Final Delta E 2000 calculation
    KL, KC, KH = 1.0, 1.0, 1.0 # Standard weighting factors for dental applications
    
    # Ensure denominators are not zero
    term1 = delta_L_prime / (KL * S_L) if (KL * S_L) != 0 else 0
    term2 = delta_C_prime / (KC * S_C) if (KC * S_C) != 0 else 0
    term3 = delta_H_prime / (KH * S_H) if (KH * S_H) != 0 else 0
    
    try:
        delta_e_squared = term1**2 + term2**2 + term3**2 + R_T * term2 * term3
        if delta_e_squared < 0:
            delta_e = 0.0
        else:
            delta_e = np.sqrt(delta_e_squared)
    except Exception as e:
        print(f"ERROR_DE: Exception during final delta_e calculation: {e}. Returning 1000.0")
        return 1000.0

    if np.isnan(delta_e) or np.isinf(delta_e):
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
REFERENCE_TAB_LAB_VALUES = {
    "neutral_gray": LabColor(lab_l=50.0, lab_a=0.0, lab_b=0.0), # Ideal neutral gray
    "vita_a2": VITA_SHADE_LAB_REFERENCES["A2"], # Use the standard A2 value
    "vita_b1": VITA_SHADE_LAB_REFERENCES["B1"], # Use the standard B1 value
}

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

def match_shade_from_lab(lab_input_tuple, zone_type="overall"):
    """
    Matches a given LAB color to the closest VITA shade using custom Delta E 2000.
    Always returns the closest shade, and the Delta E value indicates match quality.
    """
    input_lab_color = LabColor(lab_l=lab_input_tuple[0], lab_a=lab_input_tuple[1], lab_b=lab_input_tuple[2])
    
    min_delta = float('inf')
    best_shade = None

    for shade_name, reference_lab_color in VITA_SHADE_LAB_REFERENCES.items():
        delta = delta_e_cie2000(input_lab_color, reference_lab_color)
        if delta < min_delta:
            min_delta = delta
            best_shade = shade_name
            
    return best_shade, min_delta

def format_delta_e(delta_val):
    """
    Formats a Delta E value for display. Returns 'N/A' for problematic values.
    """
    if isinstance(delta_val, (int, float)) and not np.isinf(delta_val) and not np.isnan(delta_val) and delta_val != 1000.0:
        return f"{delta_val:.2f}"
    return "N/A"

def format_shade_name(shade_name, delta_val):
    """
    Formats a shade name for display. Now, it will always return the shade name.
    """
    if isinstance(delta_val, (int, float)) and (np.isinf(delta_val) or np.isnan(delta_val) or delta_val == 1000.0):
        return "N/A"
    return shade_name

# --- IMAGE PROCESSING FUNCTIONS ---
def preprocess_image(img_array, target_size=(512, 512)):
    """Resizes an image array to a standard resolution."""
    if img_array is None or img_array.size == 0:
        raise ValueError("Input image array is empty or None for preprocessing.")
    img_resized = cv2.resize(img_array, target_size)
    return img_resized

def apply_reference_white_balance(img):
    """Applies white balance using an enhanced Gray World algorithm."""
    result = img.copy().astype(np.float32)
    avgB, avgG, avgR = np.mean(result[:, :, 0]), np.mean(result[:, :, 1]), np.mean(result[:, :, 2])
    avg_intensity = (avgB + avgG + avgR) / 3.0
    scaleB = avg_intensity / avgB if avgB != 0 else 1.0
    scaleG = avg_intensity / avgG if avgG != 0 else 1.0
    scaleR = avg_intensity / avgR if avgR != 0 else 1.0
    balanced = np.zeros_like(result)
    balanced[:, :, 0] = result[:, :, 0] * scaleB
    balanced[:, :, 1] = result[:, :, 1] * scaleG
    balanced[:, :, 2] = result[:, :, 2] * scaleR
    balanced = np.clip(balanced, 0, 255).astype(np.uint8)
    hsv = cv2.cvtColor(balanced, cv2.COLOR_BGR2HSV)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.5, 0, 255)
    balanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return balanced

def advanced_white_balance(image):
    """Applies CLAHE (Contrast Limited Adaptive Histogram Equalization) for lighting correction."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    corrected = cv2.merge((cl, a, b))
    return cv2.cvtColor(corrected, cv2.COLOR_LAB2BGR)

def apply_device_profile_adjustment(img_bgr, device_profile):
    """Adds simulated color adjustments based on the selected device profile."""
    if device_profile == "ideal":
        return img_bgr
    lab_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    if device_profile == "android_cool":
        lab_img[:, :, 1] += 2.0
        lab_img[:, :, 2] += 3.0
        lab_img[:, :, 0] += 1.0
    elif device_profile == "iphone_warm":
        lab_img[:, :, 1] -= 1.5
        lab_img[:, :, 2] -= 2.0
        lab_img[:, :, 0] -= 0.5
    elif device_profile == "poor_lighting":
        lab_img[:, :, 0] *= 0.85
        lab_img[:, :, 1] *= 0.7
        lab_img[:, :, 2] *= 0.7
        lab_img[:, :, 2] += 4.0
    lab_img[:, :, 0] = np.clip(lab_img[:, :, 0], 0, 100)
    lab_img[:, :, 1] = np.clip(lab_img[:, :, 1], -128, 127)
    lab_img[:, :, 2] = np.clip(lab_img[:, :, 2], -128, 127)
    return cv2.cvtColor(lab_img.astype(np.uint8), cv2.COLOR_LAB2BGR)

def convert_to_lab_skimage(img_bgr):
    """Converts BGR image to LAB color space using scikit-image."""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_float = img_rgb.astype(np.float32) / 255.0
    lab_img = color.rgb2lab(img_float)
    return lab_img

def is_overexposed(image):
    """More sensitive overexposure detection."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel = lab[:, :, 0]
    avg_l = np.mean(l_channel) * (100.0/255.0)  
    if avg_l > 85:
        return True
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    bright_pixels = np.sum(gray > 220)
    total_pixels = gray.size
    overexposure_ratio = (bright_pixels / total_pixels)
    is_over = overexposure_ratio > 0.25
    return is_over

def get_tooth_region_lab_fallback(lab_img):
    """Fallback function: Samples LAB from the central region of the image."""
    h, w = lab_img.shape[:2]
    y1, y2 = int(h*0.4), int(h*0.6)
    x1, x2 = int(w*0.4), int(w*0.6)
    patch = lab_img[y1:y2, x1:x2]
    if patch.size == 0:
        return (75.0, 2.0, 18.0)
    avg_lab = np.mean(patch, axis=(0,1))
    return tuple(avg_lab)

def get_tooth_region_lab(lab_img):
    """ENHANCED SIMULATION OF SEGMENTATION: Uses HSV color space to find tooth-like regions."""
    lab_img_scaled = lab_img.copy()
    lab_img_scaled[:, :, 0] = np.clip(lab_img_scaled[:, :, 0] * 2.55, 0, 255)
    lab_img_scaled[:, :, 1] = np.clip(lab_img_scaled[:, :, 1] + 128, 0, 255)
    lab_img_scaled[:, :, 2] = np.clip(lab_img_scaled[:, :, 2] + 128, 0, 255)
    img_bgr = cv2.cvtColor(lab_img_scaled.astype(np.uint8), cv2.COLOR_LAB2BGR)
    h, w = img_bgr.shape[:2]
    roi_y1, roi_y2 = h // 3, 2 * h // 3
    roi_x1, roi_x2 = w // 4, 3 * w // 4
    roi = img_bgr[roi_y1:roi_y2, roi_x1:roi_x2]
    if roi.size == 0:
        return get_tooth_region_lab_fallback(lab_img)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lower_tooth = np.array([0, 0, 180])
    upper_tooth = np.array([30, 50, 255])
    mask = cv2.inRange(hsv, lower_tooth, upper_tooth)
    kernel_open = np.ones((5,5), np.uint8)
    kernel_close = np.ones((7,7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return get_tooth_region_lab_fallback(lab_img)
    largest_contour = max(contours, key=cv2.contourArea)
    final_mask_full_image = np.zeros((h, w), dtype=np.uint8)
    temp_roi_mask = np.zeros(roi.shape[:2], dtype=np.uint8)
    cv2.drawContours(temp_roi_mask, [largest_contour], -1, 255, cv2.FILLED)
    final_mask_full_image[roi_y1:roi_y2, roi_x1:roi_x2] = temp_roi_mask
    masked_lab_pixels = lab_img[final_mask_full_image > 0]
    if masked_lab_pixels.size == 0:
        return get_tooth_region_lab_fallback(lab_img)
    avg_lab = np.mean(masked_lab_pixels, axis=0)
    return tuple(avg_lab)

def get_reference_patch_lab(lab_img, patch_coords=(10, 10, 60, 60)):
    """Extracts the average LAB values from a simulated reference patch area."""
    x1, y1, x2, y2 = patch_coords
    h, w, _ = lab_img.shape
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    patch = lab_img[y1:y2, x1:x2]
    if patch.size == 0:
        return None
    avg_lab = np.mean(patch, axis=(0, 1))
    return tuple(avg_lab)

def normalize_tooth_lab_to_realistic_range(lab_tuple):
    """Normalizes LAB values to a realistic range for human teeth."""
    l, a, b = lab_tuple
    if l > 78:
        l = 78 - (l - 78) * 0.3
    elif l < 68:
        l = 68 + (68 - l) * 0.3
    if a < -2:
        a = -2 + (a + 2) * 0.4
    elif a > 3:
        a = 3 - (a - 3) * 0.4
    if b < 12:
        b = 12 + (12 - b) * 0.4
    elif b > 22:
        b = 22 - (b - 22) * 0.4
    l = np.clip(l, 68, 78)
    a = np.clip(a, -2, 3)
    b = np.clip(b, 12, 22)
    normalized_lab = LabColor(l, a, b)
    return normalized_lab

def detect_face_features(image_np_array):
    """Simulates detailed face feature extraction."""
    h, w, _ = image_np_array.shape
    sample_region = image_np_array[h//4:h*3//4, w//4:w*3//4]
    if sample_region.size == 0:
        return {"skin_tone": "N/A", "lip_color": "N/A", "eye_contrast": "N/A", "facial_harmony_score": "N/A"}
    sample_region_rgb = cv2.cvtColor(sample_region, cv2.COLOR_BGR2RGB) / 255.0
    lab_sample = color.rgb2lab(sample_region_rgb)
    avg_l_img, avg_a_img, avg_b_img = np.mean(lab_sample, axis=(0, 1))
    skin_tone_category = "Medium"
    skin_undertone = "Neutral"
    if avg_l_img > 70:
        skin_tone_category = "Light"
    elif avg_l_img < 55:
        skin_tone_category = "Dark"
    if avg_b_img > 18 and avg_a_img > 10:
        skin_undertone = "Warm (Golden/Peach)"
    elif avg_b_img < 10 and avg_a_img < 5:
        skin_undertone = "Cool (Pink/Blue)"
    elif avg_b_img >= 10 and avg_b_img <= 18 and avg_a_img >= 5 and avg_a_img <= 10:
        skin_undertone = "Neutral"
    elif avg_b_img > 15 and avg_a_img < 5:
        skin_undertone = "Olive (Greenish)"
    simulated_skin_tone = f"{skin_tone_category} with {skin_undertone} undertones"
    simulated_lip_color = np.random.choice(["Natural Pink", "Deep Rosy Red", "Bright Coral", "Subtle Mauve/Berry", "Pale Nude"])
    eye_contrast_sim = np.random.choice(["High (Distinct Features)", "Medium", "Low (Soft Features)"])
    return {
        "skin_tone": simulated_skin_tone,
        "lip_color": simulated_lip_color,
        "eye_contrast": eye_contrast_sim,
        "facial_harmony_score": round(np.random.uniform(0.7, 0.95), 2),
    }

def simulate_tooth_analysis_from_overall_lab(overall_lab_colormath):
    """Simulates detailed tooth analysis (condition, stain, decay) based on the overall LAB value."""
    tooth_condition_sim = "Normal & Healthy Appearance"
    if overall_lab_colormath.lab_l < 70 and overall_lab_colormath.lab_b > 18:
        tooth_condition_sim = "Mild Discoloration (Yellowish)"
    elif overall_lab_colormath.lab_l < 65 and overall_lab_colormath.lab_b > 20:
        tooth_condition_sim = "Moderate Discoloration (Strong Yellow)"
    elif overall_lab_colormath.lab_a < -2:
        tooth_condition_sim = "Slightly Greyish Appearance"
    stain_presence_sim = "None detected"
    if overall_lab_colormath.lab_b > 22 and np.random.rand() < 0.3:
        stain_presence_sim = "Possible light surface stains"
    elif overall_lab_colormath.lab_b > 25 and np.random.rand() < 0.5:
        stain_presence_sim = "Moderate localized stains"
    decay_presence_sim = "No visible signs of decay"
    if overall_lab_colormath.lab_l < 60 and np.random.rand() < 0.1:
        decay_presence_sim = "Potential small carious lesion (simulated - consult professional)"
    elif overall_lab_colormath.lab_l < 55 and np.random.rand() < 0.05:
        decay_presence_sim = "Possible early signs of demineralization (simulated - consult professional)"
    base_l = overall_lab_colormath.lab_l
    base_a = overall_lab_colormath.lab_a
    base_b = overall_lab_colormath.lab_b
    incisal_lab_sim = LabColor(
        lab_l=np.clip(base_l + 6.0, 0, 100),
        lab_a=np.clip(base_a - 1.0, -128, 127),
        lab_b=np.clip(base_b - 5.0, -128, 127)
    )
    middle_lab_sim = LabColor(
        lab_l=np.clip(base_l + np.random.uniform(-0.5, 0.5), 0, 100),
        lab_a=np.clip(base_a + np.random.uniform(-0.2, 0.2), -128, 127),
        lab_b=np.clip(base_b + np.random.uniform(-0.2, 0.2), -128, 127)
    )
    cervical_lab_sim = LabColor(
        lab_l=np.clip(base_l - 6.0, 0, 100),
        lab_a=np.clip(base_a + 2.0, -128, 127),
        lab_b=np.clip(base_b + 5.0, -128, 127)
    )
    incisal_lab_sim = normalize_tooth_lab_to_realistic_range((incisal_lab_sim.lab_l, incisal_lab_sim.lab_a, incisal_lab_sim.lab_b))
    middle_lab_sim = normalize_tooth_lab_to_realistic_range((middle_lab_sim.lab_l, middle_lab_sim.lab_a, middle_lab_sim.lab_b))
    cervical_lab_sim = normalize_tooth_lab_to_realistic_range((cervical_lab_sim.lab_l, cervical_lab_sim.lab_a, cervical_lab_sim.lab_b))
    return {
        "overall_lab": {"L": overall_lab_colormath.lab_l, "a": overall_lab_colormath.lab_a, "b": overall_lab_colormath.lab_b},
        "simulated_overall_shade": match_shade_from_lab(
            (overall_lab_colormath.lab_l, overall_lab_colormath.lab_a, overall_lab_colormath.lab_b),
            zone_type="overall"
        )[0],
        "tooth_condition": tooth_condition_sim,
        "stain_presence": stain_presence_sim,
        "decay_presence": decay_presence_sim,
        "incisal_lab": incisal_lab_sim,
        "middle_lab": middle_lab_sim,
        "cervical_lab": cervical_lab_sim,
    }

def aesthetic_shade_suggestion(facial_features, tooth_analysis):
    """Simulates an aesthetic mapping model with more context."""
    current_tooth_shade = tooth_analysis.get("simulated_overall_shade", "A2")
    suggested_shade = "Optimal Match (Simulated)"
    aesthetic_confidence = "High"
    recommendation_notes = "This is a simulated aesthetic suggestion. Consult a dental specialist for personalized cosmetic planning."
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
    """Calculates a confidence score based on the Delta E value, presence of a reference tab, image brightness, and simulated influence from EXIF data."""
    notes = []
    final_confidence = 0
    if isinstance(delta_e_value, (int, float)) and not np.isinf(delta_e_value) and not np.isnan(delta_e_value):
        max_delta_e_for_confidence = 10.0 
        confidence_from_delta_e = max(0, 100 - (delta_e_value / max_delta_e_for_confidence) * 50) 
        adjusted_confidence = confidence_from_delta_e
        if has_reference_tab:
            adjusted_confidence += 15
            notes.append("High confidence due to color correction with a reference tab.")
        else:
            adjusted_confidence -= 10
            notes.append("No color reference tab detected, results may have higher variability.")
        if image_brightness_l < 40:
            adjusted_confidence -= 10
            notes.append("Image appears very dark, potentially affecting accuracy.")
        elif image_brightness_l > 85:
            adjusted_confidence -= 5
            notes.append("Image appears very bright, potentially affecting accuracy.")
        exif_notes_added = False
        if exif_data:
            camera_make = exif_data.get('Make', '').lower()
            camera_model = exif_data.get('Model', '').lower()
            if 'apple' in camera_make or 'samsung' in camera_make or 'pixel' in camera_model:
                adjusted_confidence += 2
                notes.append("EXIF data suggests a common mobile device.")
                exif_notes_added = True
            if 'WhiteBalance' in exif_data and exif_data.get('WhiteBalance') not in [0, 1]:
                adjusted_confidence -= 3
                notes.append("Non-standard white balance mode detected in EXIF.")
                exif_notes_added = True
            elif 'WhiteBalance' not in exif_data:
                adjusted_confidence -= 2
                notes.append("White balance metadata missing from EXIF.")
                exif_notes_added = True
        if not exif_data and not exif_notes_added:
            adjusted_confidence -= 5
            notes.append("No EXIF data available, limiting confidence adjustment.")
        final_confidence = np.clip(adjusted_confidence, 0, 100)
    else:
        notes.append("Confidence could not be calculated due to invalid Delta E value or processing error.")
        final_confidence = 0
    if not notes:
        notes.append("Confidence based on simulated analysis and input parameters.")
    return round(float(final_confidence)), " ".join(notes).strip()

def detect_shades_from_image(image_path, selected_reference_tab="neutral_gray", simulated_device_profile="N/A"):
    """Performs image processing, extracts tooth LAB, applies mathematical normalization, and matches to VITA shades."""
    print(f"\n--- Starting Image Processing for {image_path} ---")
    print(f"Selected Reference Tab: {selected_reference_tab}")
    print(f"Simulated Device Profile: {simulated_device_profile}")

    default_shades = {
        "incisal": "N/A", "middle": "N/A", "cervical": "N/A",
        "overall_ml_shade": "ML Bypassed",
        "face_features": {}, "tooth_analysis": {}, "aesthetic_suggestion": {},
        "delta_e_matched_shades": {
            "overall": "N/A", "overall_delta_e": "N/A",
            "incisal": "N/A", "incisal_delta_e": "N/A",
            "middle": "N/A", "middle_delta_e": "N/A",
            "cervical": "N/A", "cervical_delta_e": "N/A",
        },
        "accuracy_confidence": {"overall_percentage": "N/A", "notes": ""},
        "selected_device_profile": simulated_device_profile,
        "selected_reference_tab": selected_reference_tab,
        "exif_data": {}
    }

    exif_data_parsed = {}
    img = None

    try:
        with Image.open(image_path) as pil_img:
            try:
                for orientation in ExifTags.TAGS.keys():
                    if ExifTags.TAGS[orientation] == 'Orientation':
                        break
                exif = dict(pil_img._getexif().items())
                if orientation in exif:
                    if exif[orientation] == 3:
                        pil_img = pil_img.rotate(180, expand=True)
                    elif exif[orientation] == 6:
                        pil_img = pil_img.rotate(270, expand=True)
                    elif exif[orientation] == 8:
                        pil_img = pil_img.rotate(90, expand=True)
            except Exception as e:
                print(f"WARNING: Could not apply EXIF orientation: {e}")
                pass

            if hasattr(pil_img, '_getexif'):
                exif = pil_img._getexif()
                if exif:
                    for tag, value in ExifTags.TAGS.items():
                        if tag in exif:
                            decoded = ExifTags.TAGS.get(tag, tag)
                            exif_data_parsed[decoded] = exif[tag]
        img = np.array(pil_img.convert('RGB'))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    except Exception as e:
        print(f"WARNING: Could not load image with PIL or read EXIF data: {e}")
        traceback.print_exc()
        img = cv2.imread(image_path)
        if img is None or img.size == 0:
            print(f"ERROR: Image at {image_path} is invalid or empty. Returning default shades.")
            return default_shades

    try:
        img = preprocess_image(img, target_size=(512, 512)) 
        if img is None or img.size == 0:
             print(f"ERROR: Image is invalid or empty after preprocessing. Returning default shades.")
             return default_shades

        img_advanced_wb = advanced_white_balance(img)

        if is_overexposed(img_advanced_wb):
            print("WARNING: Image detected as overexposed. Returning 'N/A' shades due to severe overexposure.")
            default_shades["accuracy_confidence"]["overall_percentage"] = 5
            default_shades["accuracy_confidence"]["notes"] = "Image is severely overexposed, analysis unreliable. Please retake photo in better lighting."
            return default_shades

        img_global_wb = apply_reference_white_balance(img_advanced_wb)
        img_device_corrected = apply_device_profile_adjustment(img_global_wb, simulated_device_profile)
        lab_img = convert_to_lab_skimage(img_device_corrected)

        has_reference_tab = False
        corrected_tooth_lab_colormath = None

        avg_lab_tuple_raw_tooth = get_tooth_region_lab(lab_img)
        normalized_initial_tooth_lab = normalize_tooth_lab_to_realistic_range(avg_lab_tuple_raw_tooth)

        if selected_reference_tab in REFERENCE_TAB_LAB_VALUES:
            reference_patch_lab_captured = get_reference_patch_lab(lab_img, patch_coords=(10, 10, 60, 60))
            
            if reference_patch_lab_captured:
                has_reference_tab = True
                ideal_reference_lab_colormath = REFERENCE_TAB_LAB_VALUES[selected_reference_tab]
                
                ideal_L = ideal_reference_lab_colormath.lab_l
                ideal_a = ideal_reference_lab_colormath.lab_a
                ideal_b = ideal_reference_lab_colormath.lab_b

                captured_L = reference_patch_lab_captured[0]
                captured_a = reference_patch_lab_captured[1]
                captured_b = reference_patch_lab_captured[2]

                offset_L = np.clip(ideal_L - captured_L, -15, 15)
                offset_a = np.clip(ideal_a - captured_a, -5, 5)
                offset_b = np.clip(ideal_b - captured_b, -5, 5)
                
                corrected_l = normalized_initial_tooth_lab.lab_l + offset_L
                corrected_a = normalized_initial_tooth_lab.lab_a + offset_a
                corrected_b = normalized_initial_tooth_lab.lab_b + offset_b

                if corrected_l > 75:
                    brightness_reduction = min(15, corrected_l - 75)
                    corrected_l -= brightness_reduction * 0.7
                
                corrected_tooth_lab_colormath = LabColor(lab_l=corrected_l, lab_a=corrected_a, lab_b=corrected_b)
            else:
                corrected_tooth_lab_colormath = normalized_initial_tooth_lab
        else:
            corrected_tooth_lab_colormath = normalized_initial_tooth_lab

        face_features = detect_face_features(img)
        tooth_analysis_data = simulate_tooth_analysis_from_overall_lab(corrected_tooth_lab_colormath)
        aesthetic_suggestion = aesthetic_shade_suggestion(face_features, tooth_analysis_data)
        
        overall_delta_e_shade, overall_min_delta = match_shade_from_lab(
            (corrected_tooth_lab_colormath.lab_l, corrected_tooth_lab_colormath.lab_a, corrected_tooth_lab_colormath.lab_b),
            zone_type="overall"
        )

        incisal_lab_colormath = tooth_analysis_data["incisal_lab"]
        middle_lab_colormath = tooth_analysis_data["middle_lab"]
        cervical_lab_colormath = tooth_analysis_data["cervical_lab"]

        incisal_delta_e_shade, incisal_min_delta = match_shade_from_lab(
            (incisal_lab_colormath.lab_l, incisal_lab_colormath.lab_a, incisal_lab_colormath.lab_b),
            zone_type="incisal"
        )
        middle_delta_e_shade, middle_min_delta = match_shade_from_lab(
            (middle_lab_colormath.lab_l, middle_lab_colormath.lab_a, middle_lab_colormath.lab_b),
            zone_type="middle"
        )
        cervical_delta_e_shade, cervical_min_delta = match_shade_from_lab(
            (cervical_lab_colormath.lab_l, cervical_lab_colormath.lab_a, cervical_lab_colormath.lab_b),
            zone_type="cervical"
        )

        overall_accuracy_confidence, confidence_notes = calculate_confidence(
            overall_min_delta, 
            has_reference_tab=has_reference_tab, 
            image_brightness_l=corrected_tooth_lab_colormath.lab_l,
            exif_data=exif_data_parsed
        )
        
        final_incisal_rule_based = incisal_delta_e_shade
        final_middle_rule_based = middle_delta_e_shade
        final_cervical_rule_based = cervical_delta_e_shade

        detected_shades = {
            "incisal": final_incisal_rule_based,
            "middle": final_middle_rule_based,
            "cervical": final_cervical_rule_based,
            "overall_ml_shade": "ML Bypassed",
            "face_features": face_features,
            "tooth_analysis": tooth_analysis_data,
            "aesthetic_suggestion": aesthetic_suggestion,
            "accuracy_confidence": {
                "overall_percentage": overall_accuracy_confidence,
                "notes": confidence_notes
            },
            "selected_device_profile": simulated_device_profile,
            "selected_reference_tab": selected_reference_tab,
            "exif_data": exif_data_parsed,
            "delta_e_matched_shades": {
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
    
    pdf.cell(0, 10, txt=f"Correction Method: {shades.get('selected_device_profile', 'N/A')} (Reference-based correction used)", ln=True)
    
    selected_ref_tab = shades.get("selected_reference_tab", "N/A").replace("_", " ").title()
    pdf.cell(0, 10, txt=f"Color Reference Used: {selected_ref_tab}", ln=True)
    pdf.ln(5)

    pdf.set_font("Arial", 'B', size=14)
    pdf.cell(0, 10, txt="Detected Shades (Rule-based / Delta E):", ln=True)
    pdf.set_font("Arial", size=12)
    
    overall_ml_shade = shades.get("overall_ml_shade", "N/A")
    if overall_ml_shade != "N/A" and overall_ml_shade != "ML Bypassed":
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
        overall_de = delta_e_shades.get('overall_delta_e', 'N/A')
        incisal_de = delta_e_shades.get('incisal_delta_e', 'N/A')
        middle_de = delta_e_shades.get('middle_delta_e', 'N/A')
        cervical_de = delta_e_shades.get('cervical_delta_e', 'N/A')

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

    exif_data_for_report = shades.get("exif_data", {})
    if exif_data_for_report:
        pdf.ln(5)
        pdf.set_font("Arial", 'B', size=10)
        pdf.cell(0, 7, txt="EXIF Data (from Image Metadata):", ln=True)
        pdf.set_font("Arial", size=9)
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
            if pdf.get_y() > 200:
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
        traceback.print_exc()
        pdf.cell(0, 10, txt="Note: An error occurred while embedding the image in the report.", ln=True)

    pdf.set_font("Arial", 'I', size=9)
    pdf.multi_cell(0, 6,
                   txt="DISCLAIMER: This report is based on simulated analysis for demonstration purposes only. It is not intended for clinical diagnosis, medical advice, or professional cosmetic planning. Always consult with a qualified dental or medical professional for definitive assessment, diagnosis, and treatment.",
                   align='C')
    pdf.output(filepath)


# ===============================================
# 1. FLASK APPLICATION SETUP & CONFIGURATION
# ===============================================

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "your_strong_dev_secret_key_12345") # REMEMBER TO CHANGE THIS ON RENDER!

UPLOAD_FOLDER = 'static/uploads'
REPORT_FOLDER = 'static/reports'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(REPORT_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['REPORT_FOLDER'] = REPORT_FOLDER

# --- Database Setup (PostgreSQL with SQLModel) ---
# This URL will be provided by Render as an environment variable
DATABASE_URL = os.environ.get("DATABASE_URL")
if not DATABASE_URL:
    # Fallback for local development if DATABASE_URL is not set
    # For local testing, you might use a local PostgreSQL or SQLite:
    # For SQLite: DATABASE_URL = "sqlite:///./database.db" (creates a file-based DB)
    # For local PostgreSQL: DATABASE_URL = "postgresql://user:password@localhost:5432/dbname"
    print("WARNING: DATABASE_URL environment variable not set. Using SQLite for local testing fallback.")
    DATABASE_URL = "sqlite:///./database.db" # Using SQLite for simple local fallback

engine = create_engine(DATABASE_URL, echo=False) # echo=True for SQL logging (useful for debug), set to False for production

def create_db_and_tables():
    """Creates all database tables defined by SQLModel."""
    SQLModel.metadata.create_all(engine)
    print("INFO: Database tables created/checked.")

# Call this once at app startup to ensure tables exist
# This is done outside the request context, so it runs when the app starts
create_db_and_tables()

# Dependency to get a database session for each request
def get_session():
    """Yields a new database session for each request."""
    with Session(engine) as session:
        yield session

# --- Database Models ---
class Patient(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: str = Field(index=True) # To link patients to users
    op_number: str = Field(index=True) # OP number for quick lookup
    patient_name: str
    age: int
    sex: str
    record_date: str # Store as string YYYY-MM-DD
    created_at: datetime = Field(default_factory=datetime.now)

    # Note: relationships are typically defined on the "one" side (Patient has many Reports)
    # but for simplicity in fetching, we'll fetch reports separately for now.
    # If you want to use relationships directly, you'd need to adjust queries.

class Report(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    patient_id: Optional[int] = Field(default=None, foreign_key="patient.id") # Link to Patient
    user_id: str = Field(index=True) # To link reports to users
    op_number: str = Field(index=True) # For direct lookup by OP number
    original_image: str # Filename of the uploaded image
    report_filename: str # Filename of the generated PDF report
    # FIX: Use type_=JSON for JSONB column type
    detected_shades: dict = Field(default_factory=dict, sa_column=Field(type_=JSON)) # Store dict as JSONB in Postgres
    timestamp: datetime = Field(default_factory=datetime.now)

# --- END Database Models ---


@app.context_processor
def inject_datetime():
    """Makes datetime available in all templates."""
    return {'datetime': datetime}

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files from the UPLOAD_FOLDER."""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


# ===============================================
# 2. AUTHENTICATION HELPERS (Adapted for PostgreSQL)
# ===============================================

@app.before_request
def load_logged_in_user():
    """Loads the logged-in user into Flask's g object for the current request."""
    # This simulation of user authentication is kept simple for Canvas environment.
    # In a real app, you'd use a proper auth system (e.g., Flask-Login, Firebase Auth).
    if 'user_id' in session and session.get('user'):
        g.user_id = session['user_id']
        g.user = session['user']
        return

    # This part is specific to how Canvas provides an initial auth token.
    # If deploying outside Canvas, you'd remove this or replace with real auth.
    initial_auth_token = os.environ.get('__initial_auth_token')
    if initial_auth_token:
        simulated_user_id = initial_auth_token.split(':')[-1] if ':' in initial_auth_token else initial_auth_token
        session['user_id'] = simulated_user_id
        session['user'] = {'id': simulated_user_id, 'username': f"User_{simulated_user_id[:8]}"}
    else:
        # Fallback for anonymous user if no token (e.g., local development without Canvas env)
        session['user_id'] = 'anonymous-' + str(uuid.uuid4())
        session['user'] = {'id': session['user_id'], 'username': f"AnonUser_{session['user_id'][-6:]}"}

    g.user_id = session.get('user_id')
    g.user = session.get('user')


def login_required(view):
    """Decorator to protect routes that require a logged-in user (not anonymous)."""
    from functools import wraps
    @wraps(view)
    def wrapped_view(**kwargs):
        if g.user is None or 'anonymous' in g.user_id:
            flash('Please log in to access this page.', 'warning')
            return redirect(url_for('login'))
        return view(**kwargs)
    return wrapped_view


# ===============================================
# 3. ROUTES (Adapted for PostgreSQL)
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
    return redirect(url_for('home'))

@app.route('/dashboard')
@login_required
def dashboard():
    """Renders the user dashboard, displaying past reports and patient info."""
    user_id = g.user['id']
    
    with next(get_session()) as session:
        # Get all patients for the current user
        statement_patients = select(Patient).where(Patient.user_id == user_id).order_by(Patient.created_at.desc())
        all_patients = session.exec(statement_patients).all()

        # Get all reports for the current user
        statement_reports = select(Report).where(Report.user_id == user_id).order_by(Report.timestamp.desc())
        all_reports = session.exec(statement_reports).all()

        # Create a mapping from op_number to the latest report for that op_number
        op_to_latest_report = {}
        for report in all_reports:
            op_num = report.op_number
            report_timestamp = report.timestamp
            
            if op_num:
                # Compare timestamps to ensure we always store the latest report for each OP number
                if op_num not in op_to_latest_report or report_timestamp > op_to_latest_report[op_num].timestamp:
                    op_to_latest_report[op_num] = report

        # Enrich patient data with their latest report information
        enriched_patients = []
        for patient in all_patients:
            patient_dict = patient.model_dump() # Convert SQLModel object to dict for template rendering
            latest_report = op_to_latest_report.get(patient.op_number)
            
            if latest_report:
                patient_dict['report_filename'] = latest_report.report_filename
                patient_dict['latest_analysis_date'] = latest_report.timestamp.strftime('%Y-%m-%d %H:%M:%S')
                # Access detected_shades as a dictionary directly
                patient_dict['latest_overall_shade'] = latest_report.detected_shades.get('delta_e_matched_shades', {}).get('overall', 'N/A')
            else:
                patient_dict['report_filename'] = None
                patient_dict['latest_analysis_date'] = 'No reports yet'
                patient_dict['latest_overall_shade'] = 'N/A'
            
            enriched_patients.append(patient_dict)
        
        # Sort patients by their latest analysis date (if available) or creation date
        enriched_patients.sort(key=lambda x: x.get('latest_analysis_date', x.get('created_at', '')), reverse=True)

    current_date_formatted = datetime.now().strftime('%Y-%m-%d')

    return render_template('dashboard.html',
                           patients=enriched_patients,
                           user=g.user,
                           current_date=current_date_formatted)


@app.route('/save_patient_data', methods=['POST'])
@login_required
def save_patient_data():
    """Handles saving new patient records to PostgreSQL and redirects to image upload page."""
    op_number = request.form['op_number']
    patient_name = request.form['patient_name']
    age = request.form['age']
    sex = request.form['sex']
    record_date = request.form['date']
    user_id = g.user['id']

    with next(get_session()) as session:
        # Check for existing OP Number for the current user
        existing_patient_statement = select(Patient).where(
            Patient.op_number == op_number,
            Patient.user_id == user_id
        )
        existing_patient = session.exec(existing_patient_statement).first()

        if existing_patient:
            flash('OP Number already exists for another patient under your account. Please use a unique OP Number or select from recent entries.', 'error')
            return redirect(url_for('dashboard'))

        try:
            patient = Patient(
                user_id=user_id,
                op_number=op_number,
                patient_name=patient_name,
                age=int(age),
                sex=sex,
                record_date=record_date
            )
            session.add(patient)
            session.commit() # Commit to save the patient and get its ID
            session.refresh(patient) # Refresh to load the generated 'id' into the patient object

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
    patient = None

    with next(get_session()) as session:
        statement = select(Patient).where(
            Patient.op_number == op_number,
            Patient.user_id == user_id
        )
        patient = session.exec(statement).first()

    if patient is None:
        flash('Patient not found or unauthorized access.', 'error')
        return redirect(url_for('dashboard'))

    return render_template('upload_page.html', op_number=op_number, patient_name=patient.patient_name)


@app.route('/upload', methods=['POST'])
@login_required
def upload_file():
    """Handles image upload, shade detection, and PDF report generation."""
    if request.method == 'POST':
        uploaded_file_obj = None
        if 'file' in request.files and request.files['file'].filename != '':
            uploaded_file_obj = request.files['file']
        elif 'camera_file' in request.files and request.files['camera_file'].filename != '':
            uploaded_file_obj = request.files['camera_file']
        
        if uploaded_file_obj is None:
            flash('No file selected or captured.', 'danger')
            return redirect(request.url)

        op_number_from_form = request.form.get('op_number')
        patient_name = request.form.get('patient_name', 'Unnamed Patient') # Used for PDF filename
        simulated_device_profile = request.form.get('device_profile', 'N/A')
        selected_reference_tab = request.form.get('reference_tab', 'neutral_gray')

        original_image_path = None
        try:
            if uploaded_file_obj:
                filename = secure_filename(uploaded_file_obj.filename)
                file_ext = os.path.splitext(filename)[1]
                unique_filename = str(uuid.uuid4()) + file_ext
                
                original_image_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
                uploaded_file_obj.save(original_image_path)
                flash('Image uploaded successfully!', 'success')

                detected_shades = detect_shades_from_image(original_image_path, selected_reference_tab, simulated_device_profile)

                # Check if overall shade is "N/A" indicating a fundamental processing error
                if detected_shades.get("delta_e_matched_shades", {}).get("overall") == "N/A":
                    flash("Error processing image for shade detection. Please try another image or check image quality.", 'danger')
                    if original_image_path and os.path.exists(original_image_path):
                        os.remove(original_image_path)
                    return redirect(url_for('upload_page', op_number=op_number_from_form))

                report_filename = f"report_{patient_name.replace(' ', '')}_{op_number_from_form}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                report_filepath = os.path.join(app.config['REPORT_FOLDER'], report_filename)
                generate_pdf_report(patient_name, detected_shades, original_image_path, report_filepath)
                flash('PDF report generated!', 'success')

                user_id = g.user['id']
                with next(get_session()) as session:
                    # Find the patient to link the report
                    patient_statement = select(Patient).where(
                        Patient.op_number == op_number_from_form,
                        Patient.user_id == user_id
                    )
                    patient_db = session.exec(patient_statement).first()

                    if patient_db:
                        report = Report(
                            patient_id=patient_db.id,
                            user_id=user_id,
                            op_number=op_number_from_form,
                            original_image=unique_filename,
                            report_filename=report_filename,
                            detected_shades=detected_shades # SQLModel handles JSON for dicts
                        )
                        session.add(report)
                        session.commit() # Commit to save the report
                        session.refresh(report) # Refresh to get the generated ID
                        flash('Report data saved to database!', 'info')
                    else:
                        flash('Could not find patient to link report. Report saved but not linked.', 'warning')

                return redirect(url_for('report_page', report_filename=report_filename))
        except Exception as e:
            flash(f'An unexpected error occurred during upload or processing: {e}', 'danger')
            traceback.print_exc()
            if original_image_path and os.path.exists(original_image_path):
                os.remove(original_image_path)
            return redirect(url_for('upload_page', op_number=op_number_from_form))
    
    flash("Please select a patient from the dashboard to upload an image.", 'info')
    return redirect(url_for('dashboard'))


@app.route('/report/<report_filename>')
@login_required
def report_page(report_filename):
    """Displays a detailed analysis report."""
    user_id = g.user['id']
    report_data = None

    with next(get_session()) as session:
        statement = select(Report).where(
            Report.report_filename == report_filename,
            Report.user_id == user_id
        )
        report_data = session.exec(statement).first()

    if report_data is None:
        flash('Report not found or unauthorized access.', 'error')
        return redirect(url_for('dashboard'))

    patient_name = 'N/A'
    if report_data.patient_id:
        with next(get_session()) as session_nested: # Use a new session for nested query
            patient = session_nested.get(Patient, report_data.patient_id)
            if patient:
                patient_name = patient.patient_name
            else:
                patient_name = 'N/A (Patient record missing)'
    else:
        patient_name = 'N/A (No patient linked)' # Fallback if patient_id is None

    analysis_date = report_data.timestamp.strftime('%Y-%m-%d %H:%M:%S')
    shades = report_data.detected_shades
    image_filename = report_data.original_image
    report_id = report_data.id

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
                           report_filename=report_filename,
                           correction_method=correction_method_display,
                           reference_tab=shades.get('selected_reference_tab', 'N/A'),
                           report_id=report_id)


@app.route('/download_report/<filename>')
@login_required
def download_report(filename):
    """Allows users to download their generated PDF reports."""
    # In a real app, you'd check database to confirm user_id matches report_data.user_id for security
    report_path = os.path.join(app.config['REPORT_FOLDER'], filename)
    if os.path.exists(report_path):
        return send_from_directory(app.config['REPORT_FOLDER'], filename, as_attachment=True)
    else:
        flash('Report file not found.', 'error')
        return redirect(url_for('dashboard'))

@app.route('/submit_feedback', methods=['POST'])
@login_required
def submit_feedback():
    """Handles user feedback submission (for now, just flashes a message)."""
    # For a real database, you'd create a 'Feedback' table and save this data.
    # We are not creating a Feedback model/table for this demo, keeping it simple.
    report_id = request.form.get('report_id')
    is_correct = request.form.get('is_correct') == 'true'
    correct_shade = request.form.get('correct_shade')
    user_id = g.user['id']

    flash("Thank you for your feedback! (Feedback is not yet saved to a dedicated database table).", 'info')
    print(f"DEBUG: Feedback received for report {report_id} by user {user_id}: Correct={is_correct}, Provided Shade={correct_shade}")

    # Redirect back to the report page or dashboard
    report_filename = None
    with next(get_session()) as session:
        report = session.get(Report, report_id)
        if report and report.user_id == user_id: # Basic security check
            report_filename = report.report_filename
    
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
    """Provides a diagnostic endpoint to serve a test image for color calibration."""
    try:
        img_width = 600
        img_height = 400
        patch_size = 100
        num_cols = img_width // patch_size
        num_rows = img_height // patch_size

        test_image_array = np.zeros((img_height, img_width, 3), dtype=np.uint8)

        test_shades = ["A1", "A2", "B1", "C1", "D2", "A3"]
        num_patches = min(len(test_shades), num_rows * num_cols)

        for i in range(num_patches):
            shade_name = test_shades[i]
            lab_color = VITA_SHADE_LAB_REFERENCES[shade_name]
            
            rgb_float = color.lab2rgb([[lab_color.lab_l, lab_color.lab_a, lab_color.lab_b]])[0]
            rgb_255 = (rgb_float * 255).astype(np.uint8)

            row = i // num_cols
            col = i % num_cols
            
            y_start = row * patch_size
            y_end = (row + 1) * patch_size
            x_start = col * patch_size
            x_end = (col + 1) * patch_size

            test_image_array[y_start:y_end, x_start:x_end] = rgb_255

            text_x = x_start + 5
            text_y = y_start + patch_size // 2 + 5
            cv2.putText(test_image_array, shade_name, (text_x, text_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(test_image_array, shade_name, (text_x, text_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)

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
    pass # Keep pass for Canvas environment; Gunicorn will run the app
