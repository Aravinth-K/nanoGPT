import os
import pickle
import requests
import numpy as np
from tqdm import tqdm
import zipfile

def download_file(url, local_path):
    """Download a file from URL with progress bar"""
    if not os.path.exists(local_path):
        print(f"Downloading {url}...")
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024 * 1024  # 1 MB
        with open(local_path, 'wb') as f:
            for data in tqdm(response.iter_content(block_size), 
                           total=total_size // block_size, 
                           unit='MB', 
                           desc='Downloading'):
                f.write(data)
        print("Download complete.")

def prepare_enwik8(data_dir='data/enwik8'):
    """Prepare the enwik8 dataset for byte-level language modeling."""
    
    # Create data directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    
    # Download the enwik8 dataset
    zip_path = os.path.join(data_dir, 'enwik8.zip')
    input_file_path = os.path.join(data_dir, 'enwik8')
    
    if not os.path.exists(input_file_path):
        # Download if needed
        download_file('http://mattmahoney.net/dc/enwik8.zip', zip_path)
        
        # Extract
        print("Extracting enwik8.zip...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        print("Extraction complete.")

    # Read the raw bytes
    with open(input_file_path, 'rb') as f:
        data = f.read()
    print(f"length of dataset in bytes: {len(data):,}")

    # Get all unique bytes that occur in this text
    unique_bytes = sorted(set(data))
    vocab_size = len(unique_bytes)
    print(f"Number of unique bytes: {vocab_size}")
    
    # Print some statistics about the bytes
    print("\nByte ranges found:")
    ascii_printable = sum(1 for b in unique_bytes if 32 <= b <= 126)
    ascii_control = sum(1 for b in unique_bytes if b < 32)
    ascii_extended = sum(1 for b in unique_bytes if b >= 128)
    print(f"ASCII printable (32-126): {ascii_printable} bytes")
    print(f"ASCII control (0-31): {ascii_control} bytes")
    print(f"Extended ASCII (128-255): {ascii_extended} bytes")

    # Create the mapping between bytes and integers
    # For bytes, we can just use the byte value as the token ID
    stoi = { b:b for b in unique_bytes }  # byte to int mapping (identity)
    itos = { b:b for b in unique_bytes }  # int to byte mapping (identity)

    # Create train/val/test splits (90%, 5%, 5% as per paper)
    n = len(data)
    train_data = data[:int(n*0.9)]
    val_data = data[int(n*0.9):int(n*0.95)]
    test_data = data[int(n*0.95):]

    # Convert to numpy arrays
    train_ids = np.frombuffer(train_data, dtype=np.uint8)
    val_ids = np.frombuffer(val_data, dtype=np.uint8)
    test_ids = np.frombuffer(test_data, dtype=np.uint8)

    print(f"train has {len(train_ids):,} tokens")
    print(f"val has {len(val_ids):,} tokens")
    print(f"test has {len(test_ids):,} tokens")

    # Save the arrays
    train_ids.tofile(os.path.join(data_dir, 'train.bin'))
    val_ids.tofile(os.path.join(data_dir, 'val.bin'))
    test_ids.tofile(os.path.join(data_dir, 'test.bin'))

    # Save the meta information
    meta = {
        'vocab_size': vocab_size,
        'itos': itos,
        'stoi': stoi,
    }
    with open(os.path.join(data_dir, 'meta.pkl'), 'wb') as f:
        pickle.dump(meta, f)

if __name__ == '__main__':
    prepare_enwik8()