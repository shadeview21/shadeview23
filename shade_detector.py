import cv2
import numpy as np
from PIL import Image, ExifTags
from skimage import color # For standard LAB conversion (used for comparison/utility)

# --- Custom LabColor class ---
# This class is used by the delta_e_cie2000 function
class LabColor:
    """A simple class to hold L*a*b* color values."""
    def __init__(self, lab_l, lab_a, lab_b):
        self.lab_l = float(lab_l)
        self.lab_a = float(lab_a)
        self.lab_b = float(lab_b)

    def __repr__(self):
        return f"LabColor(L={self.lab_l:.2f}, a={self.lab_a:.2f}, b={self.lab_b:.2f})"

# --- Complete CIE Delta E 2000 Implementation ---
# This is a robust implementation of the Delta E 2000 formula
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

    # Check for non-finite values at the start
    if any(not np.isfinite(x) for x in [L1, a1, b1, L2, a2, b2]):
        # print(f"ERROR_DE: Non-finite input LAB values detected. LAB1: ({L1:.2f}, {a1:.2f}, {b1:.2f}), LAB2: ({L2:.2f}, {a2:.2f}, {b2:.2f}). Returning 1000.0.")
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

# --- VITA Shade LAB Reference Values (from your app.py) ---
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

def load_image_with_exif(path):
    """
    Loads an image, handles EXIF orientation, and returns it as an RGB NumPy array.
    Includes fallback to OpenCV if PIL fails.
    """
    try:
        img = Image.open(path)
        
        # Get EXIF data and handle orientation
        exif_raw = img._getexif()
        if exif_raw:
            try:
                for orientation_tag in ExifTags.TAGS.keys():
                    if ExifTags.TAGS[orientation_tag] == 'Orientation':
                        break
                exif_dict = dict(exif_raw.items())
                if orientation_tag in exif_dict:
                    orientation = exif_dict[orientation_tag]
                    if orientation == 3:
                        img = img.rotate(180, expand=True)
                    elif orientation == 6:
                        img = img.rotate(270, expand=True)
                    elif orientation == 8:
                        img = img.rotate(90, expand=True)
            except Exception as e:
                print(f"WARNING: Could not apply EXIF orientation: {e}")
                pass # Continue without orientation if error
        
        img = img.convert('RGB')
        return np.array(img)
    except Exception as e:
        print(f"Failed to load or process image with PIL/EXIF: {e}. Attempting with OpenCV...")
        # Fallback: try cv2 imread directly
        img_cv = cv2.imread(path)
        if img_cv is None:
            raise RuntimeError(f"Cannot read image file at {path}. It might be corrupted or missing.")
        # OpenCV reads as BGR, convert to RGB for consistency
        return cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)

def srgb_to_lab(img_rgb):
    """
    Converts an sRGB image (0-255, uint8) to Lab color space (L:0-100, a,b: approx -128 to 127).
    This custom implementation follows the standard sRGB to XYZ to Lab conversion.
    Alternatively, for simplicity and robustness, you could use:
    `from skimage import color`
    `img_float = img_rgb.astype(np.float32) / 255.0`
    `lab = color.rgb2lab(img_float)`
    """
    # Assume img_rgb is uint8 array in 0-255 range, shape (H,W,3)
    # Convert to 0..1 normalized
    img = img_rgb.astype(np.float32) / 255.0

    # Apply gamma correction inverse (sRGB to linear RGB)
    def inv_gamma(c):
        return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    img_lin = inv_gamma(img)

    # Convert linear RGB to XYZ (D65 illuminant)
    # Standard sRGB to XYZ conversion matrix
    M = np.array([[0.4124564, 0.3575761, 0.1804375],
                  [0.2126729, 0.7151522, 0.0721750],
                  [0.0193339, 0.1191920, 0.9503041]], dtype=np.float32)
    
    shape = img_lin.shape # Keep original image shape
    img_lin_flat = img_lin.reshape(-1, 3).T # Flatten to (3, N) for matrix multiplication
    xyz = M.dot(img_lin_flat).T # Result shape (N, 3)
    
    # Normalize by reference white D65 (standard illuminant for Lab)
    white_d65 = np.array([0.95047, 1.00000, 1.08883]) # Xn, Yn, Zn for D65
    xyz_norm = xyz / white_d65

    # Apply f(t) function for Lab conversion
    delta = 6/29
    def f(t):
        return np.where(t > delta**3, np.cbrt(t), t/(3*delta**2)+4/29)
    f_xyz = f(xyz_norm)

    # Calculate L*, a*, b*
    L = 116 * f_xyz[:,1] - 16
    a = 500 * (f_xyz[:,0] - f_xyz[:,1])
    b = 200 * (f_xyz[:,1] - f_xyz[:,2])

    # Stack L, a, b channels and reshape back to original image dimensions
    lab = np.stack([L,a,b], axis=1).reshape(shape[0], shape[1], 3)
    return lab

def find_closest_shade(shades_palette, target_lab_tuple):
    """
    Finds the closest VITA shade from a palette to a target LAB color using Delta E 2000.
    Args:
        shades_palette (dict): Dictionary of VITA shade names to LabColor objects.
        target_lab_tuple (tuple): (L, a, b) values of the target color.
    Returns:
        tuple: (closest_shade_name, min_delta_e_distance).
    """
    target_lab_color = LabColor(target_lab_tuple[0], target_lab_tuple[1], target_lab_tuple[2])
    
    min_distance = float('inf')
    closest_shade_id = None
    
    for shade_name, lab_color_obj in shades_palette.items():
        dist = delta_e_cie2000(lab_color_obj, target_lab_color)
        if dist < min_distance:
            min_distance = dist
            closest_shade_id = shade_name
            
    return closest_shade_id, min_distance

def detect_shade_from_image(image_path, shades_palette):
    """
    Detects the overall shade of teeth from an image.
    Args:
        image_path (str): Path to the input image file.
        shades_palette (dict): Dictionary of VITA shade names to LabColor objects.
    Returns:
        tuple: (detected_shade_name, delta_e_distance_to_shade).
    """
    print(f"Processing image: {image_path}")
    img_rgb = load_image_with_exif(image_path)
    
    # Preprocess (resize) for consistent analysis area
    img_rgb_resized = cv2.resize(img_rgb, (512, 512)) # Resize to a standard size
    
    lab_img = srgb_to_lab(img_rgb_resized)

    # Compute average Lab over a central area (simulating tooth region)
    # This is a simplification; a real system would use image segmentation.
    h, w, _ = lab_img.shape
    # Take a central patch (e.g., 40% of height and width)
    y1, y2 = int(h*0.3), int(h*0.7)
    x1, x2 = int(w*0.3), int(w*0.7)
    
    if (y2 - y1) <= 0 or (x2 - x1) <= 0:
        raise ValueError("Image dimensions too small for central region extraction.")

    center_lab_region = lab_img[y1:y2, x1:x2]
    
    if center_lab_region.size == 0:
        raise ValueError("Central LAB region is empty. Image might be corrupted or too small.")

    center_lab_mean = center_lab_region.mean(axis=(0, 1))
    
    print(f"Average LAB of central tooth region: L={center_lab_mean[0]:.2f}, a={center_lab_mean[1]:.2f}, b={center_lab_mean[2]:.2f}")

    shade_id, distance = find_closest_shade(shades_palette, center_lab_mean)
    
    print(f"Closest shade: {shade_id} with Delta E 2000 distance {distance:.2f}")
    return shade_id, distance

# --- Example Usage ---
if __name__ == "__main__":
    # Ensure you have a test image file, e.g., 'test_image.jpg' in the same directory
    # For this example, I'll use the 'test4.jpg' you provided.
    test_image_path = "test4.jpg" 

    # Check if the test image exists
    import os
    if not os.path.exists(test_image_path):
        print(f"Error: Test image '{test_image_path}' not found.")
        print("Please ensure 'test4.jpg' is in the same directory as this script.")
        exit()

    try:
        detected_shade, delta_e = detect_shade_from_image(test_image_path, VITA_SHADE_LAB_REFERENCES)
        print("\n--- Detection Result ---")
        print(f"Detected Shade: {detected_shade}")
        print(f"Delta E 2000 to closest shade: {delta_e:.2f}")
    except Exception as e:
        print(f"\nAn error occurred during shade detection: {e}")
        import traceback
        traceback.print_exc()

