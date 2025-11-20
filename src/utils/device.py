"""
Device detection and selection utilities for optimal hardware acceleration.

This module provides automatic device detection for PyTorch-based models,
supporting CUDA (NVIDIA), MPS (Apple Silicon), and CPU fallback.
"""

import logging
import platform
from typing import Literal

logger = logging.getLogger(__name__)

DeviceType = Literal["cuda", "mps", "cpu", "auto"]


def detect_device() -> str:
    """
    Automatically detect the best available device for PyTorch operations.
    
    Priority order:
    1. CUDA (NVIDIA GPU) if available
    2. MPS (Apple Silicon GPU) if available
    3. CPU as fallback
    
    Returns:
        str: Device name ("cuda", "mps", or "cpu")
    
    Raises:
        ImportError: If PyTorch is not installed
    """
    try:
        import torch
    except ImportError as e:
        logger.error(
            "PyTorch is not installed. Install it with: "
            "pip install torch torchvision torchaudio"
        )
        raise ImportError(
            "PyTorch is required but not installed. "
            "See README.md for platform-specific installation instructions."
        ) from e
    
    # Check CUDA availability first (highest performance)
    if torch.cuda.is_available():
        device = "cuda"
        cuda_device_name = torch.cuda.get_device_name(0)
        cuda_version = torch.version.cuda
        logger.info(
            f"Using CUDA device: {cuda_device_name} (CUDA {cuda_version})"
        )
        return device
    
    # Check MPS availability (Apple Silicon)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
        system_info = f"{platform.system()} {platform.machine()}"
        logger.info(f"Using MPS device (Apple Silicon): {system_info}")
        return device
    
    # Fallback to CPU
    device = "cpu"
    logger.warning(
        "No GPU acceleration available. Using CPU. "
        "For better performance, see README.md for GPU setup instructions."
    )
    return device


def resolve_device(device_config: str) -> str:
    """
    Resolve the device string from configuration, supporting "auto" mode.
    
    Args:
        device_config: Device setting from config ("cuda", "mps", "cpu", or "auto")
    
    Returns:
        str: Resolved device name ("cuda", "mps", or "cpu")
    
    Raises:
        ValueError: If specified device is not available
        ImportError: If PyTorch is not installed
    """
    if device_config == "auto":
        return detect_device()
    
    # Validate explicitly specified device
    try:
        import torch
    except ImportError as e:
        logger.error("PyTorch is not installed")
        raise ImportError(
            "PyTorch is required. See README.md for installation instructions."
        ) from e
    
    if device_config == "cuda":
        if not torch.cuda.is_available():
            raise ValueError(
                "CUDA device requested but not available. "
                "Install PyTorch with CUDA support or set device to 'auto' or 'cpu'."
            )
        logger.info(f"Using explicitly configured CUDA device: {torch.cuda.get_device_name(0)}")
        return device_config
    
    elif device_config == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise ValueError(
                "MPS device requested but not available. "
                "Ensure you are running on Apple Silicon with macOS 12.3+ "
                "or set device to 'auto' or 'cpu'."
            )
        logger.info("Using explicitly configured MPS device (Apple Silicon)")
        return device_config
    
    elif device_config == "cpu":
        logger.info("Using CPU device (as configured)")
        return device_config
    
    else:
        raise ValueError(
            f"Invalid device configuration: '{device_config}'. "
            f"Valid options: 'cuda', 'mps', 'cpu', 'auto'"
        )


def get_optimal_batch_size(device: str, default: int = 64) -> int:
    """
    Suggest optimal batch size based on device type and available memory.
    
    Args:
        device: Device type ("cuda", "mps", or "cpu")
        default: Default batch size from configuration
    
    Returns:
        int: Recommended batch size
    """
    if device == "cpu":
        # Conservative batch size for CPU
        return min(default, 32)
    
    elif device == "mps":
        # Apple Silicon - moderate batch size
        return min(default, 64)
    
    elif device == "cuda":
        # NVIDIA GPU - can handle larger batches
        try:
            import torch
            if torch.cuda.is_available():
                # Get GPU memory in GB
                gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                
                if gpu_memory_gb < 6:
                    return min(default, 32)
                elif gpu_memory_gb < 12:
                    return min(default, 64)
                else:
                    return min(default, 128)
        except Exception as e:
            logger.warning(f"Could not determine GPU memory: {e}")
            return default
    
    return default

