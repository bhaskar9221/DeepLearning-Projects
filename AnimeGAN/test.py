try:
    import torch
except ImportError:
    import sys
    print("PyTorch is not installed. Install with 'pip install torch' to run this test.")
    sys.exit(1)

# select the available device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"Using device: {device}")
