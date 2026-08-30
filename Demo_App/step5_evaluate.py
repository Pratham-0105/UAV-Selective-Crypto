import os
import random
import time
import glob
import pickle
import numpy as np
import tifffile
import matplotlib.pyplot as plt
import numba

# ── 1. Numba GLCM Feature Extraction (Copied exactly from Step 3) ──
@numba.njit(cache=True)
def extract_tile_features_compiled(img):
    feat_array = np.zeros((1024, 15), dtype=np.float32)
    b = 0
    for r in range(32):
        for c in range(32):
            for ch in range(3):
                block = img[r*8 : (r+1)*8, c*8 : (c+1)*8, ch]
                
                # 1. Compute Shannon Entropy
                hist = np.zeros(256, dtype=np.float64)
                for i in range(8):
                    for j in range(8):
                        hist[block[i, j]] += 1
                ent = 0.0
                for i in range(256):
                    p = hist[i] / 64.0
                    if p > 0:
                        ent -= p * np.log2(p)
                
                # 2. Compute symmetric GLCM
                P = np.zeros((256, 256), dtype=np.float64)
                for i in range(8):
                    for j in range(7):
                        val1 = block[i, j]
                        val2 = block[i, j+1]
                        P[val1, val2] += 1.0
                        P[val2, val1] += 1.0
                
                total_sum = P.sum()
                if total_sum > 0:
                    P = P / total_sum
                    
                contrast = 0.0
                homogeneity = 0.0
                energy_asm = 0.0
                
                mu = 0.0
                for i in range(256):
                    row_sum = 0.0
                    for j in range(256):
                        row_sum += P[i, j]
                    mu += i * row_sum
                    
                var = 0.0
                for i in range(256):
                    row_sum = 0.0
                    for j in range(256):
                        row_sum += P[i, j]
                    var += row_sum * (i - mu) ** 2
                
                std = np.sqrt(var)
                correlation = 0.0
                
                for i in range(256):
                    for j in range(256):
                        val = P[i, j]
                        if val == 0:
                            continue
                        contrast += val * (i - j) ** 2
                        homogeneity += val / (1.0 + (i - j) ** 2)
                        energy_asm += val ** 2
                        if std > 1e-10:
                            correlation += val * (i - mu) * (j - mu) / (std * std)
                            
                energy = np.sqrt(energy_asm)
                
                feat_array[b, ch*5]     = np.float32(ent)
                feat_array[b, ch*5 + 1] = np.float32(energy)
                feat_array[b, ch*5 + 2] = np.float32(correlation)
                feat_array[b, ch*5 + 3] = np.float32(contrast)
                feat_array[b, ch*5 + 4] = np.float32(homogeneity)
            b += 1
    return feat_array

def main():
    print("=== GF2 Local Drone Simulation ===")
    
    # 1. Load Model
    model_path = os.path.join(os.path.dirname(__file__), "svm_rgb_model_tuned.pkl")
    print(f"Loading trained brain from {model_path}...")
    with open(model_path, "rb") as f:
        bundle = pickle.load(f)
        
    model = bundle["model"]
    scaler = bundle["scaler"]
    
    # 2. Pick random test image
    test_dir = os.path.join(os.path.dirname(__file__), "data_for_keras_aug", "test_images", "test")
    all_test_imgs = glob.glob(os.path.join(test_dir, "*.tif"))
    if not all_test_imgs:
        print("ERROR: No test images found!")
        return
        
    test_img_path = random.choice(all_test_imgs)
    print(f"Selected Unseen Test Image: {test_img_path}")
    
    # Load image
    img = tifffile.imread(test_img_path).squeeze()
    
    # 3. Extract Features
    print("Drone: Extracting features...")
    start_time = time.time()
    dummy = np.zeros((256, 256, 3), dtype=np.uint8)
    extract_tile_features_compiled(dummy) # Warmup numba
    
    loop_start = time.time()
    features = extract_tile_features_compiled(img)
    print(f"Drone: Extracted {features.shape[0]} blocks in {time.time() - loop_start:.4f} seconds")
    
    # 4. Predict
    print("Drone: Classifying blocks with SVM...")
    scaled_features = scaler.transform(features)
    predictions = model.predict(scaled_features)
    
    # Map predictions to a 32x32 grid
    pred_grid = predictions.reshape(32, 32)
    
    # Count totals
    counts = {0: 0, 1: 0, 2: 0}
    for label in predictions:
        counts[label] += 1
        
    print(f"Classification Results: LIB (0): {counts[0]}, MIB (1): {counts[1]}, HIB (2): {counts[2]}")
    
    # 5. Visualize
    print("Generating simulation overlay...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    
    # Plot 1: Original Image
    ax1.imshow(img)
    ax1.set_title("Original Image (Drone View)", fontsize=14)
    ax1.axis('off')
    
    # Plot 2: Overlay
    ax2.imshow(img)
    
    # Create color patches overlay
    # Red for HIB, Yellow for MIB, Blue for LIB
    overlay = np.zeros((256, 256, 4))
    for r in range(32):
        for c in range(32):
            label_int = pred_grid[r, c]
            if label_int == 0:     # LIB
                color = [0, 0, 1, 0.35]  # Blue
            elif label_int == 1:   # MIB
                color = [1, 1, 0, 0.45]  # Yellow
            elif label_int == 2:   # HIB
                color = [1, 0, 0, 0.65]  # Red
            else:
                color = [0, 0, 0, 0]
                
            overlay[r*8:(r+1)*8, c*8:(c+1)*8] = color

    ax2.imshow(overlay)
    ax2.set_title(f"Selective Encryption Map\\nRed=High(AES), Yellow=Med, Blue=Low(Skip)", fontsize=14)
    ax2.axis('off')
    
    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(__file__), "simulation_result.png")
    plt.savefig(out_path, bbox_inches='tight', dpi=150)
    print(f"✅ Simulation saved to: {out_path}")

if __name__ == "__main__":
    main()
