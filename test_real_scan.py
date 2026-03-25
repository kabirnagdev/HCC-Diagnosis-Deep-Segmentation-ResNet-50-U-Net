"""
Test the Model with REAL CT Scans
===================================
This script uses the actual CT scan files in your workspace.
"""

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from pathlib import Path

# Check if we can import the classifier
try:
    from inference import LiverCancerClassifier
except ImportError:
    print("❌ Error: inference.py not found or missing dependencies")
    print("\n📦 Install dependencies first:")
    print("   pip install torch torchvision numpy nibabel scikit-image matplotlib pillow")
    exit(1)


def test_with_real_scan():
    """Test the model with real CT scan from your workspace."""
    
    print("\n" + "="*70)
    print("🏥 TESTING WITH REAL CT SCAN")
    print("="*70)
    
    # Find CT scan files in your workspace
    ct_files = [
        Path("liver_0.nii/liver_0.nii"),
        Path("segmentation-10.nii/segmentation-10.nii"),
        Path("liver_0.nii"),
        Path("segmentation-10.nii")
    ]
    
    # Try to find an existing CT scan
    ct_path = None
    for file in ct_files:
        if file.exists():
            ct_path = file
            break
    
    if ct_path is None:
        print("\n❌ No CT scan files found in workspace!")
        print("\n📂 Looking for:")
        print("   • liver_0.nii/liver_0.nii")
        print("   • segmentation-10.nii/segmentation-10.nii")
        print("\n💡 Place a .nii file in this directory and update the path")
        return
    
    print(f"\n📂 Found CT scan: {ct_path}")
    
    # Load the CT scan
    print(f"\n📥 Loading CT volume...")
    try:
        nii = nib.load(str(ct_path))
        volume = np.rot90(np.array(nii.get_fdata()))
        print(f"   ✓ Volume shape: {volume.shape}")
        print(f"   ✓ Total slices: {volume.shape[2]}")
        print(f"   ✓ Value range: [{volume.min():.1f}, {volume.max():.1f}] HU")
    except Exception as e:
        print(f"   ❌ Error loading file: {e}")
        return
    
    # Check if model exists
    model_dir = Path('./models')
    if not model_dir.exists():
        print("\n❌ Error: './models' directory not found!")
        print("\n📥 You need to:")
        print("   1. Run Cell 25 in your Kaggle notebook")
        print("   2. Download the 'models/' folder from Kaggle Output")
        print("   3. Place it in this directory")
        return
    
    # Load the model
    print(f"\n🧠 Loading trained model...")
    try:
        classifier = LiverCancerClassifier(model_dir='./models')
    except Exception as e:
        print(f"   ❌ Error loading model: {e}")
        print("\n💡 Make sure all model files are in ./models/")
        return
    
    # Select interesting slices (middle region where liver usually is)
    print(f"\n🔍 Analyzing CT slices...")
    
    # Analyze multiple slices
    slice_indices = [
        volume.shape[2] // 4,      # 25% through volume
        volume.shape[2] // 2,      # Middle slice
        3 * volume.shape[2] // 4   # 75% through volume
    ]
    
    results = []
    for idx in slice_indices:
        ct_slice = volume[:, :, idx]
        result = classifier.predict_slice(ct_slice)
        result['slice_index'] = idx
        results.append(result)
        
        print(f"\n   Slice {idx:3d}/{volume.shape[2]}: {result['predicted_class']:12s} "
              f"({result['confidence']:5.1f}% confidence)")
    
    # Find the most interesting slice (highest tumor probability)
    tumor_probs = [r['probabilities'].get('Tumor', 0) for r in results]
    best_idx = np.argmax(tumor_probs)
    best_result = results[best_idx]
    best_slice_idx = best_result['slice_index']
    
    print(f"\n" + "="*70)
    print(f"📊 MOST SIGNIFICANT FINDINGS")
    print(f"="*70)
    print(f"\nSlice {best_slice_idx}: {best_result['predicted_class']}")
    print(f"Confidence: {best_result['confidence']:.1f}%")
    print(f"\nProbabilities:")
    for class_name, prob in best_result['probabilities'].items():
        bar = "█" * int(prob * 50)
        marker = " ⚠️" if class_name == "Tumor" and prob > 0.5 else ""
        print(f"   {class_name:12s} {prob*100:6.2f}% {bar}{marker}")
    
    # Create visualization
    print(f"\n📊 Creating visualization...")
    ct_slice = volume[:, :, best_slice_idx]
    
    try:
        classifier.visualize_prediction(ct_slice, save_path='real_scan_prediction.png')
        print(f"   ✓ Saved: real_scan_prediction.png")
    except Exception as e:
        print(f"   ⚠️  Visualization error: {e}")
        
        # Fallback: simple matplotlib visualization
        print(f"   Creating simple visualization instead...")
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.imshow(ct_slice, cmap='gray')
        ax.set_title(f"Slice {best_slice_idx}: {best_result['predicted_class']}\n"
                    f"Confidence: {best_result['confidence']:.1f}%",
                    fontsize=14, fontweight='bold')
        ax.axis('off')
        plt.tight_layout()
        plt.savefig('real_scan_simple.png', dpi=150, bbox_inches='tight')
        print(f"   ✓ Saved: real_scan_simple.png")
        plt.close()
    
    # Medical interpretation
    print(f"\n" + "="*70)
    print(f"🏥 CLINICAL INTERPRETATION")
    print(f"="*70)
    
    if best_result['predicted_class'] == 'Tumor':
        print(f"\n⚠️  WARNING: Tumor detected!")
        print(f"   • Location: Slice {best_slice_idx}/{volume.shape[2]}")
        print(f"   • Confidence: {best_result['confidence']:.1f}%")
        print(f"\n   ⚕️  Recommendation:")
        print(f"   • Further investigation required")
        print(f"   • Consult with radiologist")
        print(f"   • Consider additional imaging studies")
        
    elif best_result['predicted_class'] == 'Liver':
        print(f"\n✅ Liver tissue detected (no tumor in analyzed slices)")
        print(f"   • Best slice analyzed: {best_slice_idx}/{volume.shape[2]}")
        print(f"   • Confidence: {best_result['confidence']:.1f}%")
        print(f"\n   💡 Note: This is a limited sample. Full volume analysis recommended.")
        
    else:
        print(f"\n— Background region detected")
        print(f"   • Slice {best_slice_idx} contains minimal tissue")
        print(f"   • Try different slice indices for better analysis")
    
    print(f"\n" + "="*70)
    print(f"✅ Analysis complete!")
    print(f"="*70)
    
    return results


def analyze_full_volume():
    """Analyze ALL slices in the volume (takes longer)."""
    
    print("\n" + "="*70)
    print("🔬 FULL VOLUME ANALYSIS (All Slices)")
    print("="*70)
    
    # Find CT scan
    ct_path = Path("liver_0.nii/liver_0.nii")
    if not ct_path.exists():
        ct_path = Path("segmentation-10.nii/segmentation-10.nii")
    
    if not ct_path.exists():
        print("❌ CT scan not found!")
        return
    
    # Load volume
    print(f"\n📥 Loading {ct_path}...")
    nii = nib.load(str(ct_path))
    volume = np.rot90(np.array(nii.get_fdata()))
    
    print(f"   Volume: {volume.shape}")
    
    # Load model
    print(f"\n🧠 Loading model...")
    classifier = LiverCancerClassifier(model_dir='./models')
    
    # Analyze all slices
    print(f"\n🔍 Analyzing {volume.shape[2]} slices...")
    
    all_results = []
    for i in range(volume.shape[2]):
        ct_slice = volume[:, :, i]
        result = classifier.predict_slice(ct_slice)
        result['slice_index'] = i
        all_results.append(result)
        
        # Progress indicator
        if (i + 1) % 10 == 0:
            print(f"   Progress: {i+1}/{volume.shape[2]} slices")
    
    # Summary statistics
    print(f"\n" + "="*70)
    print(f"📊 VOLUME SUMMARY")
    print(f"="*70)
    
    tumor_slices = [r for r in all_results if r['predicted_class'] == 'Tumor']
    liver_slices = [r for r in all_results if r['predicted_class'] == 'Liver']
    bg_slices = [r for r in all_results if r['predicted_class'] == 'Background']
    
    print(f"\nTotal slices: {len(all_results)}")
    print(f"   Tumor detected:  {len(tumor_slices):3d} slices ({len(tumor_slices)/len(all_results)*100:.1f}%)")
    print(f"   Liver only:      {len(liver_slices):3d} slices ({len(liver_slices)/len(all_results)*100:.1f}%)")
    print(f"   Background:      {len(bg_slices):3d} slices ({len(bg_slices)/len(all_results)*100:.1f}%)")
    
    if tumor_slices:
        print(f"\n⚠️  TUMOR DETECTED in {len(tumor_slices)} slices:")
        for r in tumor_slices[:10]:  # Show first 10
            print(f"      Slice {r['slice_index']:3d}: {r['confidence']:.1f}% confidence")
        if len(tumor_slices) > 10:
            print(f"      ... and {len(tumor_slices) - 10} more")
    
    return all_results


if __name__ == '__main__':
    import sys
    
    print("\n🏥 LIVER CANCER DETECTION - REAL CT SCAN TEST")
    print("="*70)
    
    # Quick test with 3 slices
    print("\n[Option 1] Quick Test (3 sample slices)")
    print("[Option 2] Full Volume Analysis (all slices - slower)")
    
    choice = input("\nEnter choice (1 or 2) [default=1]: ").strip() or "1"
    
    try:
        if choice == "2":
            results = analyze_full_volume()
        else:
            results = test_with_real_scan()
        
        print("\n✅ Test completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
