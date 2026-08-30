import os
import time
import pickle
import numpy as np
import numba
import gradio as gr
from PIL import Image
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend

AES_KEY = os.urandom(32)
AES_NONCE = os.urandom(16)
CHACHA_KEY = os.urandom(32)
CHACHA_NONCE = os.urandom(16)
XOR_MASK = 0b10101010

def encrypt_aes(data_bytes):
    cipher = Cipher(algorithms.AES(AES_KEY), modes.CTR(AES_NONCE), backend=default_backend())
    encryptor = cipher.encryptor()
    return encryptor.update(data_bytes) + encryptor.finalize()

def encrypt_chacha20(data_bytes):
    cipher = Cipher(algorithms.ChaCha20(CHACHA_KEY, CHACHA_NONCE), mode=None, backend=default_backend())
    encryptor = cipher.encryptor()
    return encryptor.update(data_bytes)

def encrypt_xor(data_array):
    return np.bitwise_xor(data_array, XOR_MASK)


# ── 1. Load Model ──
# demo_app.py is inside Demo_App/, so we go up one level to reach Models/
model_path = os.path.join(os.path.dirname(__file__), "..", "Models", "svm_rgb_model_tuned.pkl")
with open(model_path, "rb") as f:
    bundle = pickle.load(f)
svm_model = bundle["model"]
scaler = bundle["scaler"]

# ── 2. Numba GLCM Feature Extraction (Ultra-Fast Parallel) ──
@numba.njit(parallel=True, cache=True)
def extract_tile_features_compiled(img):
    # Determine how many blocks
    h, w, ch = img.shape
    blocks_h = h // 8
    blocks_w = w // 8
    feat_array = np.zeros((blocks_h * blocks_w, 15), dtype=np.float32)
    
    for r in numba.prange(blocks_h):
        for c in range(blocks_w):
            b = r * blocks_w + c
            for ch_idx in range(3):
                block = img[r*8 : (r+1)*8, c*8 : (c+1)*8, ch_idx]
                
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
                
                feat_array[b, ch_idx*5]     = np.float32(ent)
                feat_array[b, ch_idx*5 + 1] = np.float32(energy)
                feat_array[b, ch_idx*5 + 2] = np.float32(correlation)
                feat_array[b, ch_idx*5 + 3] = np.float32(contrast)
                feat_array[b, ch_idx*5 + 4] = np.float32(homogeneity)
                
    return feat_array, blocks_h, blocks_w

# Warmup Numba
dummy = np.zeros((256, 256, 3), dtype=np.uint8)
_, _, _ = extract_tile_features_compiled(dummy)

# ── 3. Gradio Inference Function ──

def load_latest_drone_image():
    latest_img_path = os.path.join(os.path.dirname(__file__), 'latest_drone_capture.jpg')
    if os.path.exists(latest_img_path):
        import time
        from PIL import Image
        # Wait a tiny bit just in case it's still writing
        time.sleep(0.1)
        try:
            return Image.open(latest_img_path)
        except:
            return None
    return None

def process_image(input_img):
    if input_img is None:
        return None, None, None, "No image provided."

    img = np.array(input_img)
    if img.shape[2] == 4:
        img = img[:, :, :3]
        
    h, w, ch = img.shape
    new_h = (h // 8) * 8
    new_w = (w // 8) * 8
    img = img[:new_h, :new_w, :]

    import time
    t_start = time.time()
    
    # 1. Extract Features and Classify
    features, blocks_h, blocks_w = extract_tile_features_compiled(img)
    scaled_features = scaler.transform(features)
    predictions = svm_model.predict(scaled_features)
    pred_grid = predictions.reshape((blocks_h, blocks_w))
    
    t_class_done = time.time()
    time_classification = t_class_done - t_start
    
    overlay = np.zeros((new_h, new_w, 4), dtype=np.uint8)
    blue = [0, 100, 255, 90]
    yellow = [255, 230, 0, 110]
    red = [255, 0, 50, 150]
    counts = {0: 0, 1: 0, 2: 0}
    
    for r in range(blocks_h):
        for c in range(blocks_w):
            label_int = pred_grid[r, c]
            counts[label_int] += 1
            if label_int == 0: color = blue
            elif label_int == 1: color = yellow
            elif label_int == 2: color = red
            overlay[r*8:(r+1)*8, c*8:(c+1)*8] = color
            
    # ACTUAL ENCRYPTION PAYLOAD
    t_enc_start = time.time()
    hib_blocks, mib_blocks, lib_blocks = [], [], []
    for r in range(blocks_h):
        for c in range(blocks_w):
            label = pred_grid[r, c]
            block = img[r*8:(r+1)*8, c*8:(c+1)*8, :]
            if label == 2: hib_blocks.append(block)
            elif label == 1: mib_blocks.append(block)
            else: lib_blocks.append(block)
                
    enc_hib_blocks, enc_mib_blocks, enc_lib_blocks = [], [], []
    if hib_blocks:
        enc_hib_blocks = np.frombuffer(encrypt_aes(np.array(hib_blocks).tobytes()), dtype=np.uint8).reshape(-1, 8, 8, 3)
    if mib_blocks:
        enc_mib_blocks = np.frombuffer(encrypt_chacha20(np.array(mib_blocks).tobytes()), dtype=np.uint8).reshape(-1, 8, 8, 3)
    if lib_blocks:
        enc_lib_blocks = encrypt_xor(np.array(lib_blocks))
        
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
                
    t_enc_done = time.time()
    time_encryption = t_enc_done - t_enc_start
    
    # Transmission Simulation (assume 10 Mbps drone link, payload size in bits)
    payload_size_bits = new_h * new_w * 3 * 8
    transmission_speed_bps = 10 * 1024 * 1024 # 10 Mbps
    time_transmission = payload_size_bits / transmission_speed_bps
    
    # DECRYPTION PAYLOAD
    t_dec_start = time.time()
    dec_hib_blocks, dec_mib_blocks, dec_lib_blocks = [], [], []
    if hib_blocks:
        dec_hib_blocks = np.frombuffer(encrypt_aes(np.array(enc_hib_blocks).tobytes()), dtype=np.uint8).reshape(-1, 8, 8, 3)
    if mib_blocks:
        dec_mib_blocks = np.frombuffer(encrypt_chacha20(np.array(enc_mib_blocks).tobytes()), dtype=np.uint8).reshape(-1, 8, 8, 3)
    if lib_blocks:
        dec_lib_blocks = encrypt_xor(np.array(enc_lib_blocks))
        
    decrypted_img = np.zeros_like(img)
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
                
    t_dec_done = time.time()
    time_decryption = t_dec_done - t_dec_start
    
    # PIL conversion for Gradio
    orig_pil = Image.fromarray(img).convert("RGBA")
    overlay_pil = Image.fromarray(overlay).convert("RGBA")
    final_img = Image.alpha_composite(orig_pil, overlay_pil).convert("RGB")
    encrypted_pil = Image.fromarray(encrypted_img).convert("RGB")
    decrypted_pil = Image.fromarray(decrypted_img).convert("RGB")

    
    total = blocks_h * blocks_w
    total_time = time_classification + time_encryption + time_transmission + time_decryption
    
    stats_html = f"""
    <div style='background-color: rgba(30,30,40,0.8); padding: 20px; border-radius: 12px; border: 1px solid #444; color: white;'>
        <h3 style='margin-top: 0; color: #4cd137;'>⏱️ GF2 End-to-End Pipeline Latency</h3>
        <table style='width: 100%; font-size: 16px; margin-bottom: 15px; border-collapse: collapse;'>
            <tr style='border-bottom: 1px solid #555;'><td style='padding: 5px 0;'><b>1. ML Classification (Extract + SVM)</b></td><td style='text-align: right; color:#a5b4fc;'>{time_classification:.4f} sec</td></tr>
            <tr style='border-bottom: 1px solid #555;'><td style='padding: 5px 0;'><b>2. Selective Encryption (Drone)</b></td><td style='text-align: right; color:#ff7675;'>{time_encryption:.4f} sec</td></tr>
            <tr style='border-bottom: 1px solid #555;'><td style='padding: 5px 0;'><b>3. Simulated Transmission (10 Mbps Link)</b></td><td style='text-align: right; color:#fdcb6e;'>{time_transmission:.4f} sec</td></tr>
            <tr style='border-bottom: 1px solid #555;'><td style='padding: 5px 0;'><b>4. Payload Decryption (Ground Station)</b></td><td style='text-align: right; color:#55efc4;'>{time_decryption:.4f} sec</td></tr>
            <tr><td style='padding: 5px 0; font-size: 18px;'><b>Total Pipeline Time</b></td><td style='text-align: right; font-size: 18px; color: white;'><b>{total_time:.4f} sec</b></td></tr>
        </table>
        
        <div style='display: flex; gap: 15px;'>
            <div style='flex: 1; background: rgba(255, 0, 50, 0.2); border-left: 4px solid #ff0032; padding: 10px; border-radius: 5px;'>
                <b style='color:#ff0032; font-size:18px;'>HIB (AES-256)</b><br>
                <span style='font-size:24px;'>{counts[2]:,} blocks</span>
            </div>
            <div style='flex: 1; background: rgba(255, 230, 0, 0.2); border-left: 4px solid #ffe600; padding: 10px; border-radius: 5px;'>
                <b style='color:#ffe600; font-size:18px;'>MIB (ChaCha20)</b><br>
                <span style='font-size:24px;'>{counts[1]:,} blocks</span>
            </div>
            <div style='flex: 1; background: rgba(0, 100, 255, 0.2); border-left: 4px solid #0064ff; padding: 10px; border-radius: 5px;'>
                <b style='color:#0064ff; font-size:18px;'>LIB (XOR Mask)</b><br>
                <span style='font-size:24px;'>{counts[0]:,} blocks</span>
            </div>
        </div>
    </div>
    """
    
    return final_img, encrypted_pil, decrypted_pil, stats_html

# ── 4. Gradio UI ──
custom_css = """
body {
    background-color: #0b0f19;
    color: #e2e8f0;
}
.gradio-container {
    background: transparent !important;
}
.title-header {
    text-align: center;
    padding: 30px;
    background: linear-gradient(135deg, #1e1b4b 0%, #312e81 100%);
    border-radius: 12px;
    margin-bottom: 30px;
    border: 1px solid #4f46e5;
    box-shadow: 0 10px 25px rgba(0,0,0,0.5);
}
.title-header h1 {
    color: #ffffff;
    font-size: 2.5rem;
    font-weight: 800;
    margin: 0;
    text-shadow: 0 2px 10px rgba(0,0,0,0.5);
}
.title-header p {
    color: #a5b4fc;
    font-size: 1.2rem;
    margin-top: 10px;
}
"""

with gr.Blocks(css=custom_css) as demo:
    gr.HTML('''
    <div class="title-header">
        <h1>Ground Station Terminal</h1>
        <p>Receiving and decrypting GF2 AES/ChaCha payloads</p>
    </div>
    ''')
    

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📡 Secure Incoming Transmission (Encrypted Payload)")
            encrypted_output = gr.Image(type="pil", label="AES + ChaCha20 + XOR Encrypted Result", interactive=False)
            
        with gr.Column(scale=1):
            gr.Markdown("### 🔓 Ground Station Reconstruction (Decrypted)")
            decrypted_output = gr.Image(type="pil", label="100% Lossless Recovery", interactive=False)
            
    # Hidden components to satisfy process_image return signature
    input_image = gr.Image(visible=False)
    output_image = gr.Image(visible=False)
    
    with gr.Row():
        refresh_btn = gr.Button("📡 Poll Receiver for Latest Transmission", variant="primary", size="lg")

    with gr.Row():
        output_html = gr.HTML()
        
    # Examples
    test_dir = os.path.join(os.path.dirname(__file__), "..", "data_for_keras_aug", "test_images", "test")
    import glob
    example_imgs = glob.glob(os.path.join(test_dir, "*.tif"))
    if len(example_imgs) > 3:
        example_imgs = example_imgs[:3]
        gr.Examples(examples=example_imgs, inputs=input_image)
        
    refresh_btn.click(
        fn=load_latest_drone_image,
        inputs=[],
        outputs=[input_image]
    ).then(
        fn=process_image,
        inputs=[input_image],
        outputs=[output_image, encrypted_output, decrypted_output, output_html]
    )

if __name__ == "__main__":
    print("Starting Gradio Web Server on port 7861...")
    demo.launch(server_name="0.0.0.0", server_port=7861, share=False)
