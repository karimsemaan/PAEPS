#!/usr/bin/env python3
"""
Quick test script to verify custom models setup
"""

import sys
import warnings
warnings.filterwarnings('ignore')

def test_imports():
    """Test if all required packages are installed"""
    print("Testing package imports...")

    packages = []

    # Standard packages
    try:
        import pandas
        packages.append(("pandas", "✓"))
    except:
        packages.append(("pandas", "✗"))

    try:
        import numpy
        packages.append(("numpy", "✓"))
    except:
        packages.append(("numpy", "✗"))

    try:
        import sklearn
        packages.append(("scikit-learn", "✓"))
    except:
        packages.append(("scikit-learn", "✗"))

    # Deep learning packages
    try:
        import torch
        packages.append(("PyTorch", "✓"))
        print(f"  PyTorch version: {torch.__version__}")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"  Device: {device}")
    except:
        packages.append(("PyTorch", "✗"))

    try:
        import transformers
        packages.append(("Transformers", "✓"))
        print(f"  Transformers version: {transformers.__version__}")
    except:
        packages.append(("Transformers", "✗"))

    print("\nPackage Status:")
    for pkg, status in packages:
        print(f"  {pkg}: {status}")

    return all(status == "✓" for _, status in packages)

def test_data():
    """Test if data files exist"""
    print("\nChecking data files...")
    import os

    files_to_check = [
        "data/emails_annotated.csv",
        "notebooks/experiment_logger.py",
        "notebooks/4_custom_models.ipynb"
    ]

    all_exist = True
    for file in files_to_check:
        exists = os.path.exists(file)
        status = "✓" if exists else "✗"
        print(f"  {file}: {status}")
        all_exist = all_exist and exists

    return all_exist

def main():
    print("="*60)
    print("INTELIPS Custom Models Setup Test")
    print("="*60)

    imports_ok = test_imports()
    data_ok = test_data()

    print("\n" + "="*60)
    if imports_ok and data_ok:
        print("✅ ALL TESTS PASSED - Ready to run Notebook 4!")
        print("\nNext steps:")
        print("1. Open notebooks/4_custom_models.ipynb")
        print("2. Run all cells to train the 3 custom models:")
        print("   - Context-Aware MLP (target: 0.75-0.78 F1)")
        print("   - HCEC Model (target: 0.78-0.85 F1)")
        print("   - Fine-tuned BERT (target: 0.73-0.76 F1)")
        print("3. Models will be saved to results/models/")
        print("4. Results will be logged and visualized")
        return 0
    else:
        print("❌ Some tests failed - please check the issues above")
        return 1

if __name__ == "__main__":
    sys.exit(main())