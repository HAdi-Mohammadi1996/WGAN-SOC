import cv2
import numpy as np
import math
from scipy.ndimage import median_filter

def get_deskew_angle_hough_gradient(image):
    """
    Detects the skew angle using strictly horizontal gradients and 
    finding the longest dominant line (ignoring short noise/pores).
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    h, w = gray.shape

    # 1. Sobel Y-Direction Only
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    abs_sobel_y = np.absolute(sobel_y)
    
    # Convert to 8-bit
    sobel_8u = np.uint8(255 * abs_sobel_y / np.max(abs_sobel_y))

    # 2. Thresholding to isolate strong edges
    _, binary_edges = cv2.threshold(sobel_8u, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 3. Probabilistic Hough Line Transform
    min_length = w * 0.5
    max_gap = 50
    
    lines = cv2.HoughLinesP(binary_edges, 1, np.pi / 180, threshold=50, 
                            minLineLength=min_length, maxLineGap=max_gap)

    debug_img = cv2.cvtColor(binary_edges, cv2.COLOR_GRAY2BGR)
    
    valid_angles = []
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
            
            # Filter: Horizontal Check
            if abs(angle) > 15:
                continue

            # Filter: Border Check
            if y1 < h * 0.02 or y1 > h * 0.98:
                continue
                
            valid_angles.append(angle)
            cv2.line(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    if valid_angles:
        final_angle = np.median(valid_angles)
    else:
        print("No valid long lines found.")
        final_angle = 0.0

    return final_angle, sobel_8u, debug_img


def smooth_angles_temporal(angles, window_size=5, max_deviation=3.0):
    """
    Apply temporal smoothing to angle sequence.
    """
    angles_array = np.array(angles)
    
    # First pass: Apply median filter for temporal smoothing
    smoothed = median_filter(angles_array, size=window_size, mode='nearest')
    
    # Second pass: Detect outliers
    deviations = np.abs(angles_array - smoothed)
    outlier_flags = deviations > max_deviation
    
    # Third pass: Replace outliers with smoothed values
    final_angles = angles_array.copy()
    final_angles[outlier_flags] = smoothed[outlier_flags]
    
    return final_angles, outlier_flags


def smooth_angles_moving_average(angles, window_size=5, max_jump=2.0):
    """
    Apply moving average with jump detection.
    """
    angles_array = np.array(angles)
    smoothed = angles_array.copy()
    
    for i in range(1, len(angles_array)):
        # Check if jump is too large
        jump = abs(angles_array[i] - smoothed[i-1])
        
        if jump > max_jump:
            # Use weighted average with previous frames
            start_idx = max(0, i - window_size)
            window = smoothed[start_idx:i]
            smoothed[i] = np.mean(window)
        else:
            smoothed[i] = angles_array[i]
    
    return smoothed


def align_with_phase_correlation(img1, img2, max_shift=20):
    """
    Use phase correlation to find optimal alignment between consecutive frames.
    Returns the angle adjustment needed.
    """
    f1 = np.float32(img1)
    f2 = np.float32(img2)
    
    best_score = -1
    best_angle = 0
    
    for angle_offset in np.linspace(-5, 5, 21):
        h, w = img2.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle_offset, 1.0)
        rotated = cv2.warpAffine(f2, M, (w, h), flags=cv2.INTER_LINEAR, 
                                  borderMode=cv2.BORDER_REPLICATE)
        
        score = cv2.matchTemplate(f1, rotated, cv2.TM_CCOEFF_NORMED).max()
        
        if score > best_score:
            best_score = score
            best_angle = angle_offset
    
    return best_angle


def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, 
                              borderMode=cv2.BORDER_REPLICATE)
    return rotated


def remove_black_vignette(image_path, threshold=10):
    """
    Remove L-shaped black vignette/border from images
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None, None
        
    img_filtered = cv2.medianBlur(img, 5)
    mask = img_filtered > threshold
    
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    
    row_indices = np.where(rows)[0]
    col_indices = np.where(cols)[0]
    
    if len(row_indices) > 0 and len(col_indices) > 0:
        min_row, max_row = row_indices[0], row_indices[-1]
        min_col, max_col = col_indices[0], col_indices[-1]
        
        cropped = img[min_row:max_row+1, min_col:max_col+1]
        return img, cropped
    else:
        return img, img