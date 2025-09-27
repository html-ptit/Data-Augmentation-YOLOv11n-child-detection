# NOTE: ensure the src is >= 224
# dynamic calculate mask_value 
def compute_brightness_difference(orig_crop, child_crop):
    orig_gray = cv2.cvtColor(orig_crop, cv2.COLOR_BGR2GRAY)
    child_gray = cv2.cvtColor(child_crop, cv2.COLOR_BGR2GRAY)
    return abs(np.mean(orig_gray) - np.mean(child_gray))

def compute_color_diff(orig_crop, child_crop):
    diff = 0
    for i in range(3):
        hist_orig = cv2.calcHist([orig_crop], [i], None, [32], [0, 256])
        hist_child = cv2.calcHist([child_crop], [i], None, [32], [0, 256])
        hist_orig = cv2.normalize(hist_orig, hist_orig).flatten()
        hist_child = cv2.normalize(hist_child, hist_child).flatten()
        diff += cv2.compareHist(hist_orig, hist_child, cv2.HISTCMP_BHATTACHARYYA)
    return diff / 3

def adaptive_mask_value(orig_crop, child_crop):
    lap_var = cv2.Laplacian(cv2.cvtColor(orig_crop, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
    brightness_diff = compute_brightness_difference(orig_crop, child_crop)
    color_diff = compute_color_diff(orig_crop, child_crop)

    # Normalize and combine scores
    noise_score = np.clip(lap_var / 300, 0, 1)
    bright_score = np.clip(brightness_diff / 100, 0, 1)
    color_score = np.clip(color_diff / 0.5, 0, 1)
    print(f"noise score: {noise_score}, bright score: {bright_score}, color score: {color_score}")

    score = 0.2 * (1 - noise_score) + 0.4 * (1 - bright_score) + 0.4 * (1 - color_score)
    print(f"Calculated mask_value: {score * 255}")

    return int(np.clip(score * 255, 100, 200))  # Dynamic range control

# --- Your Original Code with Adaptation ---
# Load images
orig = cv2.imread('/content/orig.png')
child = cv2.imread('/content/child_ups-removebg-resize.png', cv2.IMREAD_UNCHANGED)  # Load with alpha
child_rgb = child[..., :3]

# Get center position
center = (orig.shape[1] // 2, orig.shape[0] // 2)

# Crop region from original image that matches child's size
h, w = child_rgb.shape[:2]

margin_ratio = 0.2  # 20% extra on each side
mh = int(h * margin_ratio)
mw = int(w * margin_ratio)

# Compute expanded crop boundaries
y1 = max(center[1] - h // 2 - mh, 0)
y2 = min(center[1] + h // 2 + mh, orig.shape[0])
x1 = max(center[0] - w // 2 - mw, 0)
x2 = min(center[0] + w // 2 + mw, orig.shape[1])

orig_crop = orig[y1:y2, x1:x2]

# Resize crop if not matching due to boundary
orig_crop = cv2.resize(orig_crop, (w, h)) if orig_crop.shape[:2] != (h, w) else orig_crop

# Compute dynamic mask value
mask_value = adaptive_mask_value(orig_crop, child_rgb)
print(f"Mask value: {mask_value}")

# Create mask using dynamic value
mask = np.ones(child_rgb.shape, dtype=np.uint8) * mask_value

# Poisson blending
output = cv2.seamlessClone(child_rgb, orig, mask, center, cv2.MIXED_CLONE)

# Save result
cv2.imwrite('/content/blended_result_ups_3metric.png', output)
