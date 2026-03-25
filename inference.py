"""
Liver Cancer Classification Inference Script
============================================
This script loads the trained model and performs inference on new CT scans.

Usage:
    python inference.py --image path/to/ct_scan.nii
"""

import json
import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
import nibabel as nib
from skimage import exposure
import matplotlib.pyplot as plt


class MobileNetClassifier(nn.Module):
    """Custom MobileNet classifier matching the trained model architecture."""
    
    def __init__(self, num_classes: int = 3):
        super().__init__()
        # Load MobileNetV2 backbone
        self.backbone = models.mobilenet_v2(weights=None)
        
        # Replace classifier to match trained model
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)


class LiverCancerClassifier:
    """Inference class for liver cancer classification."""
    
    def __init__(self, model_dir: str = './models'):
        """
        Initialize the classifier.
        
        Args:
            model_dir: Directory containing the saved model files
        """
        self.model_dir = Path(model_dir)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load configuration
        with open(self.model_dir / 'model_info.json', 'r') as f:
            self.config = json.load(f)
        
        # Load model
        self.model = self._load_model()
        self.model.eval()
        
        # Setup preprocessing
        self.transform = T.Compose([
            T.Resize((self.config['image_size'], self.config['image_size'])),
            T.ToTensor(),
            T.Normalize(
                mean=self.config['preprocessing']['normalization_mean'],
                std=self.config['preprocessing']['normalization_std']
            )
        ])
        
        print(f"✅ Model loaded successfully!")
        print(f"   Device: {self.device}")
        print(f"   Architecture: {self.config['architecture']}")
        print(f"   Test Accuracy: {self.config['metrics']['test_accuracy']:.2f}%")
    
    def _load_model(self) -> nn.Module:
        """Load the trained model."""
        # Load full model directly (safest approach for custom architectures)
        model_path = self.model_dir / 'liver_cancer_model_full.pth'
        if model_path.exists():
            try:
                model = torch.load(model_path, map_location=self.device, weights_only=False)
                print(f"   Loaded: liver_cancer_model_full.pth")
                if hasattr(model, 'to'):
                    model = model.to(self.device)
                return model
            except Exception as e:
                print(f"   Warning: {e}")
        
        # Fallback: try mobilenet_best.pth with custom architecture
        model = MobileNetClassifier(num_classes=self.config['num_classes'])
        model_path = self.model_dir / 'mobilenet_best.pth'
        if model_path.exists():
            try:
                state_dict = torch.load(model_path, map_location=self.device, weights_only=False)
                model.load_state_dict(state_dict)
                print(f"   Loaded: mobilenet_best.pth")
                return model.to(self.device)
            except Exception as e:
                print(f"   Warning: {e}")
        
        raise RuntimeError("Could not load model from any available file")
    
    def preprocess_ct_slice(self, ct_slice: np.ndarray) -> np.ndarray:
        """
        Apply HU windowing and CLAHE preprocessing.
        
        Args:
            ct_slice: Raw CT slice in HU units
            
        Returns:
            Preprocessed image
        """
        hu_min = self.config['preprocessing']['hu_window'][0]
        hu_max = self.config['preprocessing']['hu_window'][1]
        clip_limit = self.config['preprocessing']['clahe_clip_limit']
        
        # HU Windowing
        windowed = np.clip(ct_slice, hu_min, hu_max)
        windowed = (windowed - hu_min) / (hu_max - hu_min)
        
        # CLAHE
        enhanced = exposure.equalize_adapthist(windowed, clip_limit=clip_limit)
        
        return enhanced.astype(np.float32)
    
    def predict_slice(self, ct_slice: np.ndarray) -> dict:
        """
        Predict class for a single CT slice.
        
        Args:
            ct_slice: 2D numpy array of CT slice
            
        Returns:
            Dictionary containing prediction results
        """
        # Preprocess
        preprocessed = self.preprocess_ct_slice(ct_slice)
        
        # Convert to PIL Image and apply transforms
        img_pil = Image.fromarray((preprocessed * 255).astype(np.uint8))
        img_pil = img_pil.convert('RGB')  # Convert to 3 channels
        img_tensor = self.transform(img_pil).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(img_tensor)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
            pred_class = int(outputs.argmax(1).cpu().numpy()[0])
        
        return {
            'predicted_class': self.config['class_names'][pred_class],
            'predicted_class_id': pred_class,
            'probabilities': {
                name: float(prob) 
                for name, prob in zip(self.config['class_names'], probs)
            },
            'confidence': float(probs[pred_class]) * 100
        }
    
    def predict_volume(self, nii_path: str, slice_range: Tuple[int, int] = None) -> list:
        """
        Predict classes for all slices in a NIfTI volume.
        
        Args:
            nii_path: Path to NIfTI file
            slice_range: Optional tuple (start, end) to process specific slices
            
        Returns:
            List of predictions for each slice
        """
        # Load NIfTI volume
        nii = nib.load(nii_path)
        volume = np.rot90(np.array(nii.get_fdata()))
        
        # Determine slice range
        if slice_range is None:
            slice_range = (0, volume.shape[2])
        
        print(f"Processing {slice_range[1] - slice_range[0]} slices...")
        
        # Process each slice
        results = []
        for i in range(slice_range[0], slice_range[1]):
            ct_slice = volume[:, :, i]
            result = self.predict_slice(ct_slice)
            result['slice_index'] = i
            results.append(result)
        
        return results
    
    def visualize_prediction(self, ct_slice: np.ndarray, save_path: str = None):
        """
        Visualize prediction with overlay.
        
        Args:
            ct_slice: 2D CT slice
            save_path: Optional path to save visualization
        """
        # Get prediction
        result = self.predict_slice(ct_slice)
        
        # Preprocess for display
        preprocessed = self.preprocess_ct_slice(ct_slice)
        
        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original
        axes[0].imshow(preprocessed, cmap='bone')
        axes[0].set_title('Preprocessed CT Slice', fontweight='bold')
        axes[0].axis('off')
        
        # Prediction overlay
        axes[1].imshow(preprocessed, cmap='bone')
        pred_class = result['predicted_class_id']
        
        # Add colored overlay
        overlay_colors = {0: [0.5, 0.5, 0.5], 1: [0, 0, 1], 2: [1, 0, 0]}
        if pred_class > 0:
            overlay = np.zeros((*preprocessed.shape, 4))
            overlay[:, :, :3] = overlay_colors[pred_class]
            overlay[:, :, 3] = 0.3
            axes[1].imshow(overlay)
        
        axes[1].set_title(f"Prediction: {result['predicted_class']}\n"
                         f"Confidence: {result['confidence']:.1f}%",
                         fontweight='bold',
                         color=['gray', 'blue', 'red'][pred_class])
        axes[1].axis('off')
        
        # Probability bars
        classes = self.config['class_names']
        probs = [result['probabilities'][c] * 100 for c in classes]
        colors = ['gray', 'blue', 'red']
        bars = axes[2].barh(classes, probs, color=colors, alpha=0.7)
        axes[2].set_xlim([0, 100])
        axes[2].set_xlabel('Probability (%)')
        axes[2].set_title('Class Probabilities', fontweight='bold')
        
        # Highlight predicted class
        bars[pred_class].set_edgecolor('black')
        bars[pred_class].set_linewidth(2)
        
        plt.suptitle('🔬 Liver Cancer Classification Result', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"💾 Visualization saved to: {save_path}")
        
        plt.show()


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description='Liver Cancer Classification Inference')
    parser.add_argument('--image', type=str, required=True, 
                       help='Path to CT scan (NIfTI file)')
    parser.add_argument('--model-dir', type=str, default='./models',
                       help='Directory containing model files')
    parser.add_argument('--slice', type=int, default=None,
                       help='Specific slice number to analyze (optional)')
    parser.add_argument('--output', type=str, default='./prediction_result.png',
                       help='Path to save visualization')
    
    args = parser.parse_args()
    
    # Initialize classifier
    print("\n🔬 Liver Cancer Classification System")
    print("=" * 70)
    classifier = LiverCancerClassifier(model_dir=args.model_dir)
    
    # Load CT scan
    print(f"\n📂 Loading CT scan: {args.image}")
    nii = nib.load(args.image)
    volume = np.rot90(np.array(nii.get_fdata()))
    
    # Select slice
    if args.slice is None:
        slice_idx = volume.shape[2] // 2  # Middle slice
    else:
        slice_idx = args.slice
    
    ct_slice = volume[:, :, slice_idx]
    print(f"   Analyzing slice {slice_idx} of {volume.shape[2]}")
    
    # Predict
    print("\n🔍 Running inference...")
    result = classifier.predict_slice(ct_slice)
    
    # Display results
    print("\n" + "=" * 70)
    print("📊 RESULTS")
    print("=" * 70)
    print(f"Predicted Class: {result['predicted_class']}")
    print(f"Confidence: {result['confidence']:.2f}%\n")
    print("Class Probabilities:")
    for class_name, prob in result['probabilities'].items():
        print(f"   {class_name:12s}: {prob*100:6.2f}%")
    print("=" * 70)
    
    # Visualize
    print(f"\n📊 Creating visualization...")
    classifier.visualize_prediction(ct_slice, save_path=args.output)
    
    # Interpretation
    if result['predicted_class'] == 'Tumor':
        print("\n⚠️  WARNING: Tumor detected! Recommend immediate medical consultation.")
    elif result['predicted_class'] == 'Liver':
        print("\n✅ Liver tissue detected. No tumor found in this slice.")
    else:
        print("\n— Background region (no liver tissue).")


if __name__ == '__main__':
    main()
