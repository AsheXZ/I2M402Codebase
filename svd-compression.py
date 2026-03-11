import numpy as np
import matplotlib.pyplot as plt
import os

def perform_svd(matrix):
    """
    Computes SVD using the methodology from svd-auto.py
    """
    try:
        # 1. Input Validation
        A = np.asanyarray(matrix)
        if A.ndim != 2:
            return 1, "Error: Input must be a 2D matrix.", None

        # 2. Compute SVD
        # U: Left singular vectors, S: Singular values, Vh: Right singular vectors (transposed)
        U, S, Vh = np.linalg.svd(A, full_matrices=False)
        
        return 0, "Success", (U, S, Vh)

    except np.linalg.LinAlgError:
        return 2, "Error: SVD computation did not converge.", None
    except Exception as e:
        return 3, f"Error: {str(e)}", None

def compress_channel(channel_matrix, k):
    """
    Compresses a single colour channel using SVD keeping top k singular values.
    """
    code, msg, result = perform_svd(channel_matrix)
    if code != 0:
        print(f"SVD Failed for channel: {msg}")
        return channel_matrix
    
    U, S, Vh = result
    
    # Keep top k singular values
    # U is (m, r), S is (r,), Vh is (r, n)
    # We slice them to (m, k), (k,), (k, n)
    
    # Ensure k is not larger than the number of singular values
    k = min(k, len(S))
    
    U_k = U[:, :k]
    S_k = np.diag(S[:k])
    Vh_k = Vh[:k, :]
    
    # Reconstruct: A_k = U_k * S_k * Vh_k
    compressed = np.dot(U_k, np.dot(S_k, Vh_k))
    
    return compressed

def load_image(filename):
    if not os.path.exists(filename):
        print(f"Error: File {filename} not found.")
        return None
    try:
        img = plt.imread(filename)
        return img
    except Exception as e:
        print(f"Error reading image: {e}")
        return None

def main():
    image_path = "original.png"
    img = load_image(image_path)
    
    if img is None:
        # Create a dummy image if original.png doesn't exist for demonstration
        print("Creating dummy image for demonstration since original.png is missing...")
        img = np.zeros((100, 100, 3))
        img[20:60, 20:60, 0] = 1 # Red square
        img[40:80, 40:80, 1] = 1 # Green square
        # Normalize to 0-1 just in case
    
    print(f"Original Image Shape: {img.shape}")
    
    # Handle RGB vs RGBA vs Grayscale
    if img.ndim == 3:
        if img.shape[2] >= 3:
            # Separate RGB (ignore Alpha for compression if present, or maintain it)
            # For simplicity, let's just compress the first 3 channels
            channels = [img[:,:,i] for i in range(3)]
        else:
            print("Image does not have enough channels.")
            return
    else:
        # Grayscale
        channels = [img]
        
    k_values = [5, 20, 50]
    
    plt.figure(figsize=(15, 5))
    
    # Show Original
    plt.subplot(1, len(k_values) + 1, 1)
    plt.title("Original")
    plt.imshow(img)
    plt.axis('off')
    
    for i, k in enumerate(k_values):
        print(f"Compressing with k={k}...")
        compressed_channels = []
        for channel in channels:
            compressed_c = compress_channel(channel, k)
            compressed_channels.append(compressed_c)
            
        # Stack channels back
        if len(compressed_channels) == 3:
            compressed_img = np.stack(compressed_channels, axis=2)
            # Clip values to valid range [0, 1] for floats or [0, 255] for ints
            if img.dtype == np.float32 or img.dtype == float:
                compressed_img = np.clip(compressed_img, 0.0, 1.0)
            else:
                compressed_img = np.clip(compressed_img, 0, 255).astype(img.dtype)
                
            # If original had alpha, maybe append it back (untouched)
            if img.shape[2] == 4:
                alpha = img[:,:,3]
                compressed_img = np.dstack((compressed_img, alpha))
                
        else:
            compressed_img = compressed_channels[0]
            
        plt.subplot(1, len(k_values) + 1, i + 2)
        plt.title(f"k={k}")
        plt.imshow(compressed_img, cmap='gray' if img.ndim==2 else None)
        plt.axis('off')
        
    output_file = "compression_comparison.png"
    plt.savefig(output_file)
    print(f"Saved comparison to {output_file}")
    plt.close()

if __name__ == "__main__":
    main()
