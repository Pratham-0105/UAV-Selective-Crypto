"""
step6_encryptor.py
===================
This module handles the actual cryptographic payload encryption.
Based on the selective encryption map (HIB, MIB, LIB), this script will:
- Apply AES-256 (CTR) to HIB blocks (High Security)
- Apply ChaCha20 to MIB blocks (Medium Security)
- Apply XOR masking to LIB blocks (Low Security)
"""

import os
import time
import pickle
import numpy as np
import tifffile
import numba
import random
import glob
from PIL import Image

# Cryptography modules
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend

# ── 1. Load Model ──
model_path = os.path.join(os.path.dirname(__file__), "..", "Models", "svm_rgb_model_tuned.pkl")
with open(model_path, "rb") as f:
    bundle = pickle.load(f)
svm_model = bundle["model"]
scaler = bundle["scaler"]

# ── 2. Numba GLCM Feature Extraction (Ultra-Fast Parallel) ──
@numba.njit(parallel=True, cache=True)
def extract_tile_features_compiled(img):
    h, w, ch = img.shape
    blocks_h = h // 8
    blocks_w = w // 8
    feat_array = np.zeros((blocks_h * blocks_w, 15), dtype=np.float32)
    
    for r in numba.prange(blocks_h):
        for c in range(blocks_w):
            b = r * blocks_w + c
            for ch_idx in range(3):
                block = img[r*8 : (r+1)*8, c*8 : (c+1)*8, ch_idx]
                
                hist = np.zeros(256, dtype=np.float64)
                for i in range(8):
                    for j in range(8):
                        hist[block[i, j]] += 1
                ent = 0.0
                for i in range(256):
                    p = hist[i] / 64.0
                    if p > 0:
                        ent -= p * np.log2(p)
                
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
                
                feat_array[b, ch_idx*5]     = np.float32(ent)
                feat_array[b, ch_idx*5 + 1] = np.float32(energy)
                feat_array[b, ch_idx*5 + 2] = np.float32(correlation)
                feat_array[b, ch_idx*5 + 3] = np.float32(contrast)
                feat_array[b, ch_idx*5 + 4] = np.float32(homogeneity)
                
    return feat_array, blocks_h, blocks_w

# ── 3. Encryption Handlers ──
# Keys and nonces (In reality, negotiated via secure key exchange)
AES_KEY = os.urandom(32)   # 256-bit key
AES_NONCE = os.urandom(16) # 128-bit nonce for CTR

CHACHA_KEY = os.urandom(32)
CHACHA_NONCE = os.urandom(16)

# XOR mask key (just 1 byte repeated, for extreme speed)
XOR_MASK = 0b10101010  # 170

def encrypt_aes(data_bytes):
    cipher = Cipher(algorithms.AES(AES_KEY), modes.CTR(AES_NONCE), backend=default_backend())
    encryptor = cipher.encryptor()
    return encryptor.update(data_bytes) + encryptor.finalize()

def encrypt_chacha20(data_bytes):
    cipher = Cipher(algorithms.ChaCha20(CHACHA_KEY, CHACHA_NONCE), mode=None, backend=default_backend())
    encryptor = cipher.encryptor()
    return encryptor.update(data_bytes)

def encrypt_xor(data_array):
    # Extremely fast numpy bitwise XOR
    return np.bitwise_xor(data_array, XOR_MASK)

# ── 4. Main Process ──
def process_selective_encryption(img_path):
    print(f"Loading image: {img_path}")
    img = tifffile.imread(img_path).squeeze()
    if len(img.shape) == 2:
        img = np.stack((img,)*3, axis=-1)
        
    # Crop to multiple of 8
    h, w, ch = img.shape
    new_h = (h // 8) * 8
    new_w = (w // 8) * 8
    img = img[:new_h, :new_w, :]
    
    print("Extracting features and classifying...")
    t0 = time.time()
    
    # Predict blocks
    features, blocks_h, blocks_w = extract_tile_features_compiled(img)
    scaled_features = scaler.transform(features)
    predictions = svm_model.predict(scaled_features)
    pred_grid = predictions.reshape((blocks_h, blocks_w))
    
    t1 = time.time()
    print(f"Classification took {t1 - t0:.4f}s")
    
    print("Applying Selective Encryption...")
    
    encrypted_img = np.zeros_like(img)
    
    # We will measure encryption time
    enc_start = time.time()
    
    # 1. Create masks for each label
    hib_mask = (pred_grid == 2)
    mib_mask = (pred_grid == 1)
    lib_mask = (pred_grid == 0)
    
    counts = {
        2: np.sum(hib_mask),
        1: np.sum(mib_mask),
        0: np.sum(lib_mask)
    }
    
    # We reshape the image into a flat array of blocks: (1024, 8, 8, 3)
    # Actually, we can use numpy block extraction, but standard reshaping is tricky because of strides.
    # Instead, we will extract them into a list, concatenate, encrypt, and put them back.
    
    hib_blocks = []
    mib_blocks = []
    lib_blocks = []
    
    # Gather
    for r in range(blocks_h):
        for c in range(blocks_w):
            label = pred_grid[r, c]
            block = img[r*8:(r+1)*8, c*8:(c+1)*8, :]
            if label == 2:
                hib_blocks.append(block)
            elif label == 1:
                mib_blocks.append(block)
            else:
                lib_blocks.append(block)
                
    # Encrypt in bulk (massively faster in Python than looping ciphers)
    enc_hib_blocks = []
    if hib_blocks:
        hib_data = np.array(hib_blocks).tobytes()
        enc_hib_data = encrypt_aes(hib_data)
        enc_hib_blocks = np.frombuffer(enc_hib_data, dtype=np.uint8).reshape(-1, 8, 8, 3)
        
    enc_mib_blocks = []
    if mib_blocks:
        mib_data = np.array(mib_blocks).tobytes()
        enc_mib_data = encrypt_chacha20(mib_data)
        enc_mib_blocks = np.frombuffer(enc_mib_data, dtype=np.uint8).reshape(-1, 8, 8, 3)
        
    enc_lib_blocks = []
    if lib_blocks:
        lib_data = np.array(lib_blocks)
        enc_lib_blocks = encrypt_xor(lib_data)
        
    # Scatter back
    encrypted_img = np.zeros_like(img)
    h_idx, m_idx, l_idx = 0, 0, 0
    
    for r in range(blocks_h):
        for c in range(blocks_w):
            label = pred_grid[r, c]
            if label == 2:
                encrypted_img[r*8:(r+1)*8, c*8:(c+1)*8, :] = enc_hib_blocks[h_idx]
                h_idx += 1
            elif label == 1:
                encrypted_img[r*8:(r+1)*8, c*8:(c+1)*8, :] = enc_mib_blocks[m_idx]
                m_idx += 1
            else:
                encrypted_img[r*8:(r+1)*8, c*8:(c+1)*8, :] = enc_lib_blocks[l_idx]
                l_idx += 1

    enc_end = time.time()
    enc_time = enc_end - enc_start
    print(f"Encryption completed in {enc_time:.4f}s!")
    print(f"Blocks processed - HIB: {counts[2]}, MIB: {counts[1]}, LIB: {counts[0]}")
    
    # For comparison, let's time full AES encryption of the entire image
    full_enc_start = time.time()
    _ = encrypt_aes(img.tobytes())
    full_enc_time = time.time() - full_enc_start
    
    print(f"==================================================")
    print(f"PERFORMANCE REPORT:")
    print(f"Full AES-256 (baseline) : {full_enc_time:.4f}s")
    print(f"Selective Encryption    : {enc_time:.4f}s")
    print(f"Speedup                 : {full_enc_time/enc_time:.2f}x faster!")
    print(f"==================================================")
    
    # Save output
    out_path = os.path.join(os.path.dirname(__file__), "encrypted_payload.png")
    Image.fromarray(encrypted_img).save(out_path)
    
    # Save the side-by-side for human viewing
    combined = np.hstack((img, encrypted_img))
    Image.fromarray(combined).save(os.path.join(os.path.dirname(__file__), "encrypted_image.png"))
    
    # Save transmission metadata (Map + Keys) for Receiver (step7)
    map_path = os.path.join(os.path.dirname(__file__), "encryption_map.npy")
    np.save(map_path, pred_grid)
    
    keys_path = os.path.join(os.path.dirname(__file__), "crypto_keys.pkl")
    with open(keys_path, "wb") as f:
        pickle.dump({
            "AES_KEY": AES_KEY,
            "AES_NONCE": AES_NONCE,
            "CHACHA_KEY": CHACHA_KEY,
            "CHACHA_NONCE": CHACHA_NONCE,
            "XOR_MASK": XOR_MASK
        }, f)
        
    print(f"Transmitting Payload: saved {out_path}, {map_path}, {keys_path}")

if __name__ == "__main__":
    # Warmup numba
    dummy = np.zeros((256, 256, 3), dtype=np.uint8)
    extract_tile_features_compiled(dummy)
    
    # Pick random test image
    test_dir = os.path.join(os.path.dirname(__file__), "..", "data_for_keras_aug", "test_images", "test")
    all_test_imgs = glob.glob(os.path.join(test_dir, "*.tif"))
    if not all_test_imgs:
        print("ERROR: No test images found!")
    else:
        test_img_path = random.choice(all_test_imgs)
        process_selective_encryption(test_img_path)
