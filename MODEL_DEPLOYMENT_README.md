# 🏥 Liver Cancer Classification Model - Deployment Guide

## 📦 Model Package

Your trained MobileNet-V2 liver cancer classification model is ready for deployment!

### Model Performance
- **Architecture**: MobileNet-V2 (Transfer Learning)
- **Classes**: Background, Liver, Tumor (HCC)
- **Input**: 224x224 preprocessed CT slices
- **Preprocessing**: HU Windowing [-75, 150] + CLAHE

## 🚀 Quick Start

### Step 1: Download Model Files from Kaggle

After running the notebook on Kaggle, download these files from the `models/` directory:

```
models/
├── liver_cancer_model.pth          # Model weights only
├── liver_cancer_model_full.pth     # Complete model (recommended)
├── model_info.json                 # Configuration & metrics
└── config.pkl                      # Config object
```

**To download from Kaggle:**
1. In your Kaggle notebook, run Cell 21 (model saving cell)
2. Click on the Output tab → Files
3. Download the entire `models/` folder
4. Place it in your project directory

### Step 2: Install Dependencies

```bash
pip install torch torchvision numpy nibabel scikit-image matplotlib pillow
```

### Step 3: Run Inference

#### Option A: Command Line

```bash
# Analyze a CT scan
python inference.py --image path/to/scan.nii --output result.png

# Analyze a specific slice
python inference.py --image scan.nii --slice 50 --output slice_50.png
```

#### Option B: Python Script

```python
from inference import LiverCancerClassifier
import nibabel as nib
import numpy as np

# Initialize classifier
classifier = LiverCancerClassifier(model_dir='./models')

# Load CT scan
nii = nib.load('path/to/scan.nii')
volume = np.rot90(np.array(nii.get_fdata()))
ct_slice = volume[:, :, 50]  # Select slice 50

# Get prediction
result = classifier.predict_slice(ct_slice)

print(f"Prediction: {result['predicted_class']}")
print(f"Confidence: {result['confidence']:.2f}%")
print(f"Probabilities: {result['probabilities']}")

# Visualize
classifier.visualize_prediction(ct_slice, save_path='output.png')
```

#### Option C: Process Entire Volume

```python
# Analyze all slices in a volume
results = classifier.predict_volume('scan.nii', slice_range=(20, 80))

# Count tumor detections
tumor_slices = [r for r in results if r['predicted_class'] == 'Tumor']
print(f"Tumor detected in {len(tumor_slices)} slices")
```

## 📊 Understanding the Output

The model returns:

```json
{
    "predicted_class": "Tumor",
    "predicted_class_id": 2,
    "probabilities": {
        "Background": 0.05,
        "Liver": 0.15,
        "Tumor": 0.80
    },
    "confidence": 80.0
}
```

- **Background (0)**: No liver tissue present
- **Liver (1)**: Healthy liver tissue
- **Tumor (2)**: Hepatocellular Carcinoma detected

## 🔧 Integration Examples

### Web API (Flask)

```python
from flask import Flask, request, jsonify
from inference import LiverCancerClassifier
import nibabel as nib
import numpy as np

app = Flask(__name__)
classifier = LiverCancerClassifier(model_dir='./models')

@app.route('/predict', methods=['POST'])
def predict():
    # Get uploaded file
    file = request.files['ct_scan']
    file.save('temp.nii')
    
    # Load and predict
    nii = nib.load('temp.nii')
    volume = np.rot90(np.array(nii.get_fdata()))
    
    # Process middle slice
    slice_idx = volume.shape[2] // 2
    result = classifier.predict_slice(volume[:, :, slice_idx])
    
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
```

### Desktop GUI (Tkinter)

```python
import tkinter as tk
from tkinter import filedialog
from inference import LiverCancerClassifier

class LiverCancerGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Liver Cancer Detector")
        self.classifier = LiverCancerClassifier()
        
        # Add file selection button
        btn = tk.Button(self.root, text="Select CT Scan", 
                       command=self.analyze)
        btn.pack(padx=20, pady=20)
        
        # Result label
        self.result_label = tk.Label(self.root, text="", font=('Arial', 14))
        self.result_label.pack(padx=20, pady=20)
    
    def analyze(self):
        # File dialog
        filename = filedialog.askopenfilename(
            filetypes=[("NIfTI files", "*.nii *.nii.gz")]
        )
        
        if filename:
            # Load and predict
            nii = nib.load(filename)
            volume = np.rot90(np.array(nii.get_fdata()))
            ct_slice = volume[:, :, volume.shape[2]//2]
            
            result = self.classifier.predict_slice(ct_slice)
            
            # Display result
            self.result_label.config(
                text=f"{result['predicted_class']}\n"
                     f"Confidence: {result['confidence']:.1f}%"
            )
    
    def run(self):
        self.root.mainloop()

# Run application
app = LiverCancerGUI()
app.run()
```

## 📁 Model Files Explained

| File | Size | Description | When to Use |
|------|------|-------------|-------------|
| `liver_cancer_model.pth` | ~9 MB | Model weights only | When you want to load into custom architecture |
| `liver_cancer_model_full.pth` | ~14 MB | Complete model | **Recommended** - Easiest to load |
| `model_info.json` | <1 KB | Configuration & metrics | For understanding model setup |
| `config.pkl` | <1 KB | Python Config object | For advanced customization |

## ⚙️ Preprocessing Pipeline

Your model expects images preprocessed with:

1. **HU Windowing**: Clip to [-75, 150] → Normalize to [0, 1]
2. **CLAHE**: Contrast enhancement with clip_limit=0.01
3. **Resize**: 224x224 pixels
4. **Normalization**: 
   - Mean: [0.485, 0.456, 0.406]
   - Std: [0.229, 0.224, 0.225]

The `inference.py` script handles this automatically!

## 🎯 Performance Metrics

Check `model_info.json` for:
- Test Accuracy
- Sensitivity (Tumor Recall)
- Dice Coefficient
- Per-class metrics

## 🚨 Important Notes

### Medical Disclaimer
⚠️ This model is for **research and educational purposes only**. It should NOT be used for actual medical diagnosis without:
- Validation by certified radiologists
- Clinical trials and regulatory approval
- Integration into certified medical software systems

### Best Practices
1. **Always visualize**: Use `visualize_prediction()` to see what the model "sees"
2. **Check confidence**: Low confidence (<60%) predictions should be reviewed
3. **Multiple slices**: Analyze multiple slices, not just one
4. **Context matters**: Consider patient history and other diagnostic data

## 🔄 Retraining the Model

To retrain on new data:

1. Update the dataset paths in Cell 4 of the notebook
2. Adjust hyperparameters in Cell 3 (Config class)
3. Run all cells in sequence
4. Download the new model files

## 📝 Customization

### Change Input Size
```python
# In Config class (Cell 3)
IMAGE_SIZE = 256  # Default: 224
```

### Adjust HU Window
```python
# For different tissue types
HU_MIN = -100  # Default: -75
HU_MAX = 200   # Default: 150
```

### Use Different Backbone
```python
# In model creation cell
model = models.resnet18(pretrained=True)  # Instead of MobileNet-V2
```

## 📧 Support

If you encounter issues:
1. Check that all model files are in the same directory
2. Verify CT scan format (must be NIfTI: .nii or .nii.gz)
3. Ensure PyTorch version compatibility (tested on PyTorch 2.0+)

## 📜 License

This model and code are provided as-is for educational purposes.

## 🎓 Citation

If you use this model in research, please cite:
```
Liver Tumor Segmentation Dataset
https://www.kaggle.com/datasets/andrewmvd/liver-tumor-segmentation
```

---

**Made with ❤️ for medical AI research**
