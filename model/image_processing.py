import cv2
import numpy as np

def gray_world_white_balance(img):
    """
    Applies Gray World Algorithm for white balancing an image.
    Args:
        img (numpy.ndarray): The input image in BGR format.
    Returns:
        numpy.ndarray: The white-balanced image in BGR format.
    """
    result = img.copy().astype(np.float32) # Convert to float32 for calculations
    
    # Calculate average intensity for each channel
    avgB = np.mean(result[:, :, 0])
    avgG = np.mean(result[:, :, 1])
    avgR = np.mean(result[:, :, 2])
    
    # Calculate overall average gray value
    avgGray = (avgB + avgG + avgR) / 3

    # Apply scaling factor to each channel
    result[:, :, 0] = np.minimum(result[:, :, 0] * (avgGray / avgB), 255)
    result[:, :, 1] = np.minimum(result[:, :, 1] * (avgGray / avgG), 255)
    result[:, :, 2] = np.minimum(result[:, :, 2] * (avgGray / avgR), 255)

    return result.astype(np.uint8) # Convert back to uint8 for image display/saving


def correct_lighting(img_np_array):
    """
    Applies CLAHE (Contrast Limited Adaptive Histogram Equalization) for lighting correction.
    Args:
        img_np_array (numpy.ndarray): The input image (NumPy array, BGR format).
    Returns:
        numpy.ndarray: The corrected image (NumPy array, BGR format).
    """
    # Convert to LAB color space
    img_lab = cv2.cvtColor(img_np_array, cv2.COLOR_BGR2LAB)
    
    # Split channels
    l, a, b = cv2.split(img_lab)

    # Apply CLAHE to L-channel
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l_eq = clahe.apply(l)

    # Merge channels back
    img_lab_eq = cv2.merge((l_eq, a, b))

    # Convert back to BGR color space
    img_corrected = cv2.cvtColor(img_lab_eq, cv2.COLOR_LAB2BGR)
    return img_corrected

# This section below is for independent testing of this module if needed,
# but it's commented out because it requires an example image.
# You don't need to uncomment this for the Flask app to work.
# if __name__ == "__main__":
#     # Create a dummy image for testing (a red square)
#     # In a real test, you'd load an actual image file.
#     dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
#     dummy_img[:, :] = (0, 0, 255) # Blue, Green, Red (BGR)

#     # Test white balance
#     wb_img = gray_world_white_balance(dummy_img)
#     cv2.imwrite("test_wb.jpg", wb_img)
#     print("Test white balanced image saved as test_wb.jpg")

#     # Test lighting correction
#     corrected_img = correct_lighting(dummy_img)
#     cv2.imwrite("test_corrected.jpg", corrected_img)
#     print("Test corrected image saved as test_corrected.jpg")