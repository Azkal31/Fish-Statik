"""
Test script for marine image classifier - standalone version
"""
import os
import sys

print("Python version:", sys.version)
print("Python path:", sys.path)

try:
    import numpy as np
    print("✅ NumPy version:", np.__version__)
except ImportError as e:
    print("❌ NumPy import failed:", e)

try:
    import tensorflow as tf
    print("✅ TensorFlow version:", tf.__version__)
except ImportError as e:
    print("❌ TensorFlow import failed:", e)

try:
    from PIL import Image
    print("✅ Pillow/PIL imported successfully")
except ImportError as e:
    print("❌ Pillow import failed:", e)

# Test model file existence
model_path = "model_tensorflow/keras_model.h5"
labels_path = "model_tensorflow/labels.txt"

if os.path.exists(model_path):
    print(f"✅ Model file found: {model_path}")
else:
    print(f"❌ Model file not found: {model_path}")

if os.path.exists(labels_path):
    print(f"✅ Labels file found: {labels_path}")
    with open(labels_path, 'r') as f:
        lines = f.readlines()
    print(f"   Found {len(lines)} labels")
else:
    print(f"❌ Labels file not found: {labels_path}")