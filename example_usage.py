"""
Simple Example: Load and Use Liver Cancer Classification Model
================================================================

This script demonstrates the basic usage of the trained model.
"""

import sys
from pathlib import Path
import numpy as np
import torch

# Check if model files exist
model_dir = Path('./models')
if not model_dir.exists():
    print("❌ Error: './models' directory not found!")
    print("\n📥 Please download the model files from Kaggle:")
    print("   1. Run Cell 21 in your Kaggle notebook")
    print("   2. Go to Output → Files tab")
    print("   3. Download the 'models/' folder")
    print("   4. Place it in this directory")
    sys.exit(1)

# Import the classifier
try:
    from inference import LiverCancerClassifier
    import nibabel as nib
except ImportError as e:
    print(f"❌ Error: {e}")
    print("\n📦 Install required packages:")
    print("   pip install torch torchvision numpy nibabel scikit-image matplotlib pillow")
    sys.exit(1)


def example_single_slice():
    """Example 1: Classify a single CT slice"""
    print("\n" + "="*70)
    print("EXAMPLE 1: Single Slice Classification")
    print("="*70)
    
    # Initialize classifier
    print("\n📥 Loading model...")
    classifier = LiverCancerClassifier(model_dir='./models')
    
    # Option 1: If you have a NIfTI file
    # Uncomment and modify this if you have a real CT scan
    """
    nii_path = 'path/to/your/scan.nii'
    nii = nib.load(nii_path)
    volume = np.rot90(np.array(nii.get_fdata()))
    
    # Select middle slice
    slice_idx = volume.shape[2] // 2
    ct_slice = volume[:, :, slice_idx]
    """
    
    # Option 2: Create synthetic data for testing (remove this in production)
    print("\n⚠️  Using synthetic test data (replace with real CT scan)")
    ct_slice = np.random.randn(512, 512) * 50 + 30  # Simulated HU values
    
    # Predict
    print("\n🔍 Running inference...")
    result = classifier.predict_slice(ct_slice)
    
    # Display results
    print("\n" + "="*70)
    print("📊 PREDICTION RESULTS")
    print("="*70)
    print(f"\n✨ Predicted Class: {result['predicted_class']}")
    print(f"📈 Confidence: {result['confidence']:.2f}%")
    print(f"\n📊 Class Probabilities:")
    for class_name, prob in result['probabilities'].items():
        bar = "█" * int(prob * 50)
        print(f"   {class_name:12s} {prob*100:6.2f}% {bar}")
    
    # Visualize (optional - comment out if running headless)
    try:
        print("\n📊 Creating visualization...")
        classifier.visualize_prediction(ct_slice, save_path='example_prediction.png')
    except Exception as e:
        print(f"⚠️  Visualization skipped: {e}")
    
    return result


def example_batch_processing():
    """Example 2: Process multiple slices from a volume"""
    print("\n" + "="*70)
    print("EXAMPLE 2: Batch Processing (Volume)")
    print("="*70)
    
    # Initialize classifier
    classifier = LiverCancerClassifier(model_dir='./models')
    
    # Uncomment if you have a real NIfTI file
    """
    print("\n📥 Loading CT volume...")
    results = classifier.predict_volume('path/to/scan.nii', slice_range=(20, 80))
    
    # Analyze results
    tumor_slices = [r for r in results if r['predicted_class'] == 'Tumor']
    liver_slices = [r for r in results if r['predicted_class'] == 'Liver']
    
    print(f"\n📊 Volume Analysis:")
    print(f"   Total slices processed: {len(results)}")
    print(f"   Tumor detected in: {len(tumor_slices)} slices")
    print(f"   Liver tissue in: {len(liver_slices)} slices")
    
    if tumor_slices:
        print(f"\n⚠️  WARNING: Tumor detected in slices:")
        for r in tumor_slices[:5]:  # Show first 5
            print(f"      Slice {r['slice_index']}: {r['confidence']:.1f}% confidence")
    """
    
    print("\n⚠️  Replace with real CT scan path to use this example")


def example_model_info():
    """Example 3: Display model information"""
    print("\n" + "="*70)
    print("EXAMPLE 3: Model Information")
    print("="*70)
    
    import json
    
    # Load model info
    with open('./models/model_info.json', 'r') as f:
        info = json.load(f)
    
    print(f"\n🧠 Model Details:")
    print(f"   Architecture: {info['architecture']}")
    print(f"   Input Size: {info['image_size']}x{info['image_size']}")
    print(f"   Classes: {', '.join(info['class_names'])}")
    
    print(f"\n📈 Performance Metrics:")
    print(f"   Test Accuracy: {info['metrics']['test_accuracy']:.2f}%")
    print(f"   Sensitivity: {info['metrics']['test_sensitivity']:.2f}%")
    print(f"   Dice Score: {info['metrics']['test_dice']:.2f}%")
    
    print(f"\n⚙️  Preprocessing:")
    print(f"   HU Window: {info['preprocessing']['hu_window']}")
    print(f"   CLAHE Clip: {info['preprocessing']['clahe_clip_limit']}")
    
    print(f"\n🎓 Training Settings:")
    print(f"   Epochs: {info['training']['epochs']}")
    print(f"   Learning Rate: {info['training']['learning_rate']}")
    print(f"   Batch Size: {info['training']['batch_size']}")


def example_custom_preprocessing():
    """Example 4: Custom preprocessing pipeline"""
    print("\n" + "="*70)
    print("EXAMPLE 4: Custom Preprocessing")
    print("="*70)
    
    classifier = LiverCancerClassifier(model_dir='./models')
    
    # Create custom CT data
    ct_slice = np.random.randn(512, 512) * 50 + 30
    
    # Step-by-step preprocessing
    print("\n🔧 Preprocessing steps:")
    
    # 1. HU Windowing
    print("   1. Applying HU windowing...")
    hu_min, hu_max = -75, 150
    windowed = np.clip(ct_slice, hu_min, hu_max)
    windowed = (windowed - hu_min) / (hu_max - hu_min)
    print(f"      Range: [{windowed.min():.3f}, {windowed.max():.3f}]")
    
    # 2. CLAHE
    print("   2. Applying CLAHE...")
    from skimage import exposure
    enhanced = exposure.equalize_adapthist(windowed, clip_limit=0.01)
    print(f"      Range: [{enhanced.min():.3f}, {enhanced.max():.3f}]")
    
    # 3. Predict
    print("   3. Running inference...")
    result = classifier.predict_slice(ct_slice)
    print(f"      ✓ Prediction: {result['predicted_class']} ({result['confidence']:.1f}%)")


def main():
    """Run all examples"""
    print("\n" + "="*70)
    print("🏥 LIVER CANCER CLASSIFICATION - USAGE EXAMPLES")
    print("="*70)
    
    try:
        # Example 1: Basic single slice prediction
        example_single_slice()
        
        # Example 2: Model information
        example_model_info()
        
        # Example 3: Custom preprocessing
        example_custom_preprocessing()
        
        # Example 4: Batch processing (commented - needs real data)
        # example_batch_processing()
        
        print("\n" + "="*70)
        print("✅ Examples completed successfully!")
        print("="*70)
        print("\n💡 Next steps:")
        print("   1. Replace synthetic data with real CT scans")
        print("   2. Modify the code for your specific use case")
        print("   3. Integrate into your application")
        print("\n📖 See MODEL_DEPLOYMENT_README.md for more details")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
