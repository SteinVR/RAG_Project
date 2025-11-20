#!/usr/bin/env python3
"""Device detection CLI utility for verifying GPU acceleration setup."""

import sys


def main():
    """Check available compute devices and provide setup recommendations."""
    try:
        from src.utils.device import detect_device
        import torch
        
        print("=" * 60)
        print("Device Detection")
        print("=" * 60)
        
        device = detect_device()
        
        print(f"\n✓ Active device: {device}")
        print(f"  PyTorch version: {torch.__version__}")
        
        if device == "cuda":
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA version: {torch.version.cuda}")
        elif device == "mps":
            import platform
            print(f"  Platform: {platform.system()} {platform.machine()}")
        
        print("\n" + "=" * 60)
        print("Update config/settings.yaml:")
        print(f"  embeddings:")
        print(f"    device: \"{device}\"")
        print("=" * 60)
        
    except ImportError as e:
        print(f"Error: {e}", file=sys.stderr)
        print("\nInstall PyTorch first:", file=sys.stderr)
        print("  macOS M1/M2/M3: uv pip install torch torchvision torchaudio")
        print("  NVIDIA GPU: uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124")
        sys.exit(1)


if __name__ == "__main__":
    main()

