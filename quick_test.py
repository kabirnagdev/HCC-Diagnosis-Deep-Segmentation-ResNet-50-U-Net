"""Quick test of the model on real CT scan data"""

from inference import LiverCancerClassifier
import nibabel as nib
import numpy as np

print("Loading model...")
classifier = LiverCancerClassifier(model_dir='./models')

print("\nLoading CT scan...")
nii = nib.load('liver_0.nii/liver_0.nii')
volume = np.rot90(np.array(nii.get_fdata()))
print(f"Volume shape: {volume.shape}")
print(f"Total slices: {volume.shape[2]}")

# Test on 3 different slices
slices_to_test = [volume.shape[2]//4, volume.shape[2]//2, 3*volume.shape[2]//4]

print("\n" + "="*70)
print("TESTING ON REAL CT SCAN SLICES")  
print("="*70)

for idx in slices_to_test:
    ct_slice = volume[:, :, idx]
    result = classifier.predict_slice(ct_slice)
    
    print(f"\nSlice {idx}: {result['predicted_class']:12s} ({result['confidence']:.1f}% confidence)")
    print("   Probabilities:")
    for cls, prob in result['probabilities'].items():
        bar = "#" * int(prob * 30)
        print(f"      {cls:12s}: {prob*100:6.2f}% {bar}")

print("\n" + "="*70)
print("Model is working on real CT scan data!")
print("="*70)

# Save a visualization of the best prediction
print("\nSaving visualization...")
best_slice_idx = volume.shape[2] // 2
ct_slice = volume[:, :, best_slice_idx]

# Save without showing (to avoid blocking)
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
classifier.visualize_prediction(ct_slice, save_path='test_output.png')
print("\nDone! Open 'test_output.png' to see the visualization.")
