"""
step7_decryptor.py
===================
Simulates the ground station receiver.
1. Receives the encrypted payload (image).
2. Receives the transmission metadata (encryption map + keys).
3. Reverses the cryptographic algorithms.
4. Outputs the fully reconstructed original image perfectly intact.
"""

import os
import pickle
import numpy as np
import time
from PIL import Image

# Cryptography modules
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend

def decrypt_aes(data_bytes, key, nonce):
    # CTR mode decryption is mathematically identical to encryption (it just XORs the stream)
    cipher = Cipher(algorithms.AES(key), modes.CTR(nonce), backend=default_backend())
    decryptor = cipher.decryptor()
    return decryptor.update(data_bytes) + decryptor.finalize()

def decrypt_chacha20(data_bytes, key, nonce):
    cipher = Cipher(algorithms.ChaCha20(key, nonce), mode=None, backend=default_backend())
    decryptor = cipher.decryptor()
    return decryptor.update(data_bytes)

def decrypt_xor(data_array, mask):
    # XORing again with the same mask reverses it perfectly
    return np.bitwise_xor(data_array, mask)

def main():
    print("=== Ground Station Decryption System ===")
    
    # 1. Load Transmitted Files
    payload_path = os.path.join(os.path.dirname(__file__), "encrypted_payload.png")
    map_path = os.path.join(os.path.dirname(__file__), "encryption_map.npy")
    keys_path = os.path.join(os.path.dirname(__file__), "crypto_keys.pkl")
    
    if not os.path.exists(payload_path):
        print("Error: No payload found. Run step6_encryptor.py first.")
        return
        
    print("Receiving encrypted payload...")
    encrypted_img = np.array(Image.open(payload_path).convert("RGB"))
    h, w, ch = encrypted_img.shape
    blocks_h = h // 8
    blocks_w = w // 8
    
    print("Loading Secure Metadata (Map & Keys)...")
    pred_grid = np.load(map_path)
    with open(keys_path, "rb") as f:
        keys = pickle.load(f)
        
    AES_KEY = keys["AES_KEY"]
    AES_NONCE = keys["AES_NONCE"]
    CHACHA_KEY = keys["CHACHA_KEY"]
    CHACHA_NONCE = keys["CHACHA_NONCE"]
    XOR_MASK = keys["XOR_MASK"]
    
    print("Reversing Selective Cryptography...")
    t0 = time.time()
    
    # 2. Batch Processing for Speed (just like encryptor)
    hib_blocks, mib_blocks, lib_blocks = [], [], []
    
    # Gather
    for r in range(blocks_h):
        for c in range(blocks_w):
            label = pred_grid[r, c]
            block = encrypted_img[r*8:(r+1)*8, c*8:(c+1)*8, :]
            if label == 2:
                hib_blocks.append(block)
            elif label == 1:
                mib_blocks.append(block)
            else:
                lib_blocks.append(block)
                
    # Decrypt in bulk
    dec_hib_blocks = []
    if hib_blocks:
        hib_data = np.array(hib_blocks).tobytes()
        dec_hib_data = decrypt_aes(hib_data, AES_KEY, AES_NONCE)
        dec_hib_blocks = np.frombuffer(dec_hib_data, dtype=np.uint8).reshape(-1, 8, 8, 3)
        
    dec_mib_blocks = []
    if mib_blocks:
        mib_data = np.array(mib_blocks).tobytes()
        dec_mib_data = decrypt_chacha20(mib_data, CHACHA_KEY, CHACHA_NONCE)
        dec_mib_blocks = np.frombuffer(dec_mib_data, dtype=np.uint8).reshape(-1, 8, 8, 3)
        
    dec_lib_blocks = []
    if lib_blocks:
        lib_data = np.array(lib_blocks)
        dec_lib_blocks = decrypt_xor(lib_data, XOR_MASK)
        
    # Scatter back
    decrypted_img = np.zeros_like(encrypted_img)
    h_idx, m_idx, l_idx = 0, 0, 0
    
    for r in range(blocks_h):
        for c in range(blocks_w):
            label = pred_grid[r, c]
            if label == 2:
                decrypted_img[r*8:(r+1)*8, c*8:(c+1)*8, :] = dec_hib_blocks[h_idx]
                h_idx += 1
            elif label == 1:
                decrypted_img[r*8:(r+1)*8, c*8:(c+1)*8, :] = dec_mib_blocks[m_idx]
                m_idx += 1
            else:
                decrypted_img[r*8:(r+1)*8, c*8:(c+1)*8, :] = dec_lib_blocks[l_idx]
                l_idx += 1

    t1 = time.time()
    print(f"Decryption completed perfectly in {t1-t0:.4f}s!")
    
    # Save the output
    out_path = os.path.join(os.path.dirname(__file__), "decrypted_image.png")
    
    # Let's save a side-by-side: Encrypted -> Decrypted for human viewing
    combined = np.hstack((encrypted_img, decrypted_img))
    Image.fromarray(combined).save(out_path)
    print(f"Saved Reconstructed Image to: {out_path}")

if __name__ == "__main__":
    main()
