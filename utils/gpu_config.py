"""
GPU CONFIGURATION & VERIFICATION
=================================
Configures TensorFlow to use your RTX 3050 GPU efficiently.

Run this first to verify your GPU is detected:
    python utils/gpu_config.py

Supports both CUDA and DirectML backends.
"""

import os

# Suppress TF info logs (keep warnings/errors)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Try to load DirectML plugin for Windows GPU acceleration
# This must be imported BEFORE tensorflow
try:
    import tensorflow_directml_plugin  # noqa: F401
    os.environ['TF_USE_DIRECTML'] = '1'
    print("🔶 DirectML plugin loaded - using GPU acceleration")
except ImportError:
    pass

import tensorflow as tf


def configure_gpu(memory_growth=True, memory_limit_mb=None):
    """
    Configure GPU for training.

    Args:
        memory_growth: If True, allocate GPU memory dynamically (recommended)
                       If False, allocate full GPU memory upfront
        memory_limit_mb: If set, cap GPU memory usage (e.g., 3072 for 3GB)
                        Useful if you want to use the PC for other tasks while training
    """
    # Check if using DirectML
    use_directml = os.environ.get('TF_USE_DIRECTML', '0') == '1'
    
    gpus = tf.config.list_physical_devices('GPU')

    if not gpus:
        print("⚠️  NO GPU DETECTED — falling back to CPU")
        print("   Training will be slow.")
        print("   To fix:")
        print("   1. Install CUDA 11.2 + cuDNN 8.1 (for TF 2.10)")
        print("   2. Or install: pip install tensorflow-directml-plugin")
        return False

    try:
        # DirectML doesn't support memory growth - skip GPU config for it
        if not use_directml:
            for gpu in gpus:
                if memory_growth:
                    tf.config.experimental.set_memory_growth(gpu, True)
                if memory_limit_mb:
                    tf.config.set_logical_device_configuration(
                        gpu,
                        [tf.config.LogicalDeviceConfiguration(memory_limit=memory_limit_mb)]
                    )
        
        if use_directml:
            print(f"✅ DirectML GPU configured: {len(gpus)} device(s)")
        else:
            print(f"✅ GPU configured: {len(gpus)} device(s)")
        
        for i, gpu in enumerate(gpus):
            print(f"   [{i}] {gpu.name}")
        return True
    except RuntimeError as e:
        print(f"❌ GPU config failed: {e}")
        return False


def verify_gpu():
    """Detailed GPU diagnostic info."""
    print("\n" + "=" * 60)
    print("  GPU DIAGNOSTIC REPORT")
    print("=" * 60)

    print(f"\n🔧 TensorFlow version: {tf.__version__}")

    # Check DirectML
    use_directml = os.environ.get('TF_USE_DIRECTML', '0') == '1'
    if use_directml:
        print("🔶 Using DirectML backend (Windows GPU acceleration)")
    
    # Built with CUDA?
    print(f"🔧 Built with CUDA   : {tf.test.is_built_with_cuda()}")
    print(f"🔧 Built with GPU    : {tf.test.is_built_with_gpu_support()}")

    # List devices
    physical = tf.config.list_physical_devices()
    print(f"\n📱 Physical devices ({len(physical)}):")
    for d in physical:
        print(f"   - {d.device_type}: {d.name}")

    gpus = tf.config.list_physical_devices('GPU')

    if not gpus:
        print("\n❌ NO GPU FOUND")
        print("\nTroubleshooting steps:")
        print("  1. Verify NVIDIA drivers: nvidia-smi (in terminal)")
        print("  2. CUDA 11.2 must be installed for TensorFlow 2.10")
        print("  3. cuDNN 8.1 must be installed")
        print("  4. Or use DirectML: pip install tensorflow-directml-plugin")
        print("\nFor Windows with RTX 3050, the easiest fix is:")
        print("    pip install tensorflow-directml-plugin")
        print("    (works with any TF 2.x, no CUDA install needed)")
        return False

    print(f"\n✅ {len(gpus)} GPU(s) detected:")
    for i, gpu in enumerate(gpus):
        print(f"   [{i}] {gpu.name}")
        try:
            details = tf.config.experimental.get_device_details(gpu)
            if details:
                print(f"       Compute capability: {details.get('compute_capability', 'unknown')}")
                print(f"       Device name: {details.get('device_name', 'unknown')}")
        except Exception:
            pass

    # Run small computation to confirm GPU works
    print("\n🧪 Running test computation on GPU...")
    try:
        with tf.device('/GPU:0'):
            a = tf.random.normal([1000, 1000])
            b = tf.random.normal([1000, 1000])
            c = tf.matmul(a, b)
            _ = c.numpy()
        print("✅ GPU test computation succeeded")
    except Exception as e:
        print(f"❌ GPU test failed: {e}")
        return False

    backend = "DirectML" if use_directml else "CUDA"
    print("\n" + "=" * 60)
    print(f"  🚀 READY TO TRAIN ON GPU ({backend})")
    print("=" * 60 + "\n")
    return True


if __name__ == "__main__":
    configure_gpu(memory_growth=True)
    verify_gpu()