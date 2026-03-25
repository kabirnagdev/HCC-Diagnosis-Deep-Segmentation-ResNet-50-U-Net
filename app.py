"""
Simple Liver Cancer Detection App
==================================
Just run this and select your CT scan file!
"""

import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Import our classifier
from inference import LiverCancerClassifier
import nibabel as nib
from PIL import Image


class LiverCancerApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("🏥 Liver Cancer Detection")
        self.root.geometry("900x700")
        self.root.configure(bg='#1a1a2e')
        
        # Load the model
        print("Loading model...")
        self.classifier = LiverCancerClassifier(model_dir='./models')
        
        self.volume = None
        self.current_slice = 0
        
        self.setup_ui()
        
    def setup_ui(self):
        # Title
        title = tk.Label(self.root, text="🏥 Liver Cancer Detection System",
                        font=('Arial', 20, 'bold'), fg='#00d4ff', bg='#1a1a2e')
        title.pack(pady=20)
        
        # Buttons frame
        btn_frame = tk.Frame(self.root, bg='#1a1a2e')
        btn_frame.pack(pady=10)
        
        # Upload button
        self.upload_btn = tk.Button(btn_frame, text="📂 Select CT Scan (.nii)", 
                                   command=self.upload_nii,
                                   font=('Arial', 14), bg='#3498db', fg='white',
                                   padx=20, pady=10, cursor='hand2')
        self.upload_btn.pack(side=tk.LEFT, padx=10)
        
        # Upload PNG/JPG button
        self.upload_img_btn = tk.Button(btn_frame, text="🖼️ Select Image (PNG/JPG)", 
                                       command=self.upload_image,
                                       font=('Arial', 14), bg='#9b59b6', fg='white',
                                       padx=20, pady=10, cursor='hand2')
        self.upload_img_btn.pack(side=tk.LEFT, padx=10)
        
        # Slice slider frame (hidden initially)
        self.slider_frame = tk.Frame(self.root, bg='#1a1a2e')
        
        self.slice_label = tk.Label(self.slider_frame, text="Slice: 0/0",
                                   font=('Arial', 12), fg='white', bg='#1a1a2e')
        self.slice_label.pack()
        
        self.slice_slider = tk.Scale(self.slider_frame, from_=0, to=100,
                                    orient=tk.HORIZONTAL, length=400,
                                    command=self.on_slice_change,
                                    bg='#16213e', fg='white', 
                                    troughcolor='#3498db', highlightthickness=0)
        self.slice_slider.pack(pady=5)
        
        # Analyze button
        self.analyze_btn = tk.Button(self.slider_frame, text="🔍 Analyze Current Slice",
                                    command=self.analyze_slice,
                                    font=('Arial', 12, 'bold'), bg='#27ae60', fg='white',
                                    padx=15, pady=8, cursor='hand2')
        self.analyze_btn.pack(pady=10)
        
        # Result frame
        self.result_frame = tk.Frame(self.root, bg='#0f3460', padx=20, pady=20)
        
        self.result_label = tk.Label(self.result_frame, text="",
                                    font=('Arial', 16, 'bold'), fg='white', bg='#0f3460',
                                    justify=tk.LEFT)
        self.result_label.pack()
        
        # Figure for displaying image
        self.fig_frame = tk.Frame(self.root, bg='#1a1a2e')
        self.fig_frame.pack(pady=10, fill=tk.BOTH, expand=True)
        
        # Status bar
        self.status = tk.Label(self.root, text="Ready - Select a CT scan to analyze",
                              font=('Arial', 10), fg='#95a5a6', bg='#1a1a2e')
        self.status.pack(side=tk.BOTTOM, pady=10)
        
    def upload_nii(self):
        """Open file dialog to select NIfTI file."""
        file_path = filedialog.askopenfilename(
            title="Select CT Scan",
            filetypes=[
                ("NIfTI files", "*.nii *.nii.gz"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            self.load_nii(file_path)
    
    def upload_image(self):
        """Open file dialog to select PNG/JPG image."""
        file_path = filedialog.askopenfilename(
            title="Select CT Image",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            self.load_image(file_path)
    
    def load_nii(self, file_path):
        """Load NIfTI volume."""
        try:
            self.status.config(text=f"Loading: {Path(file_path).name}...")
            self.root.update()
            
            nii = nib.load(file_path)
            self.volume = np.rot90(np.array(nii.get_fdata()))
            
            # Update slider
            self.slice_slider.config(to=self.volume.shape[2] - 1)
            self.slice_slider.set(self.volume.shape[2] // 2)
            self.current_slice = self.volume.shape[2] // 2
            
            # Show slider and controls
            self.slider_frame.pack(pady=10)
            self.result_frame.pack(pady=10, fill=tk.X, padx=20)
            
            self.slice_label.config(text=f"Slice: {self.current_slice}/{self.volume.shape[2]-1}")
            self.status.config(text=f"✅ Loaded: {Path(file_path).name} ({self.volume.shape[2]} slices)")
            
            # Show first analysis
            self.analyze_slice()
            
        except Exception as e:
            messagebox.showerror("Error", f"Could not load file:\n{str(e)}")
            self.status.config(text=f"❌ Error loading file")
    
    def load_image(self, file_path):
        """Load a single image (PNG/JPG)."""
        try:
            self.status.config(text=f"Loading: {Path(file_path).name}...")
            self.root.update()
            
            # Load image
            img = Image.open(file_path).convert('L')  # Convert to grayscale
            img_array = np.array(img).astype(np.float32)
            
            # Normalize to HU-like range for preprocessing
            img_array = (img_array / 255.0) * 225 - 75  # Map to [-75, 150] HU range
            
            # Create single-slice "volume"
            self.volume = img_array[:, :, np.newaxis]
            
            # Update slider
            self.slice_slider.config(to=0)
            self.slice_slider.set(0)
            self.current_slice = 0
            
            # Show controls
            self.slider_frame.pack(pady=10)
            self.result_frame.pack(pady=10, fill=tk.X, padx=20)
            
            self.slice_label.config(text=f"Image: {Path(file_path).name}")
            self.status.config(text=f"✅ Loaded: {Path(file_path).name}")
            
            # Analyze
            self.analyze_slice()
            
        except Exception as e:
            messagebox.showerror("Error", f"Could not load image:\n{str(e)}")
            self.status.config(text=f"❌ Error loading image")
    
    def on_slice_change(self, value):
        """Handle slice slider change."""
        self.current_slice = int(value)
        if self.volume is not None:
            total = self.volume.shape[2] - 1
            self.slice_label.config(text=f"Slice: {self.current_slice}/{total}")
    
    def analyze_slice(self):
        """Analyze the current slice."""
        if self.volume is None:
            return
        
        self.status.config(text="🔍 Analyzing...")
        self.root.update()
        
        ct_slice = self.volume[:, :, self.current_slice]
        result = self.classifier.predict_slice(ct_slice)
        
        # Update result display
        pred = result['predicted_class']
        conf = result['confidence']
        probs = result['probabilities']
        
        # Color based on prediction
        if pred == 'Tumor':
            color = '#e74c3c'
            emoji = '⚠️'
            warning = "\n⚠️ TUMOR DETECTED - Consult a physician!"
        elif pred == 'Liver':
            color = '#3498db'
            emoji = '✅'
            warning = "\n✅ Liver tissue detected, no tumor found"
        else:
            color = '#95a5a6'
            emoji = '—'
            warning = ""
        
        result_text = f"""
{emoji} Prediction: {pred}
📊 Confidence: {conf:.1f}%

Probabilities:
  • Background: {probs['Background']*100:.1f}%
  • Liver: {probs['Liver']*100:.1f}%
  • Tumor: {probs['Tumor']*100:.1f}%
{warning}
"""
        self.result_label.config(text=result_text, fg=color)
        
        # Display the slice
        self.display_slice(ct_slice, result)
        
        self.status.config(text=f"✅ Analysis complete: {pred} ({conf:.1f}%)")
    
    def display_slice(self, ct_slice, result):
        """Display the CT slice with prediction overlay."""
        # Clear previous figure
        for widget in self.fig_frame.winfo_children():
            widget.destroy()
        
        # Create matplotlib figure
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        fig.patch.set_facecolor('#1a1a2e')
        
        # Preprocess for display
        preprocessed = self.classifier.preprocess_ct_slice(ct_slice)
        
        # Original
        axes[0].imshow(preprocessed, cmap='bone')
        axes[0].set_title('CT Slice', color='white', fontweight='bold')
        axes[0].axis('off')
        
        # With overlay
        axes[1].imshow(preprocessed, cmap='bone')
        
        pred = result['predicted_class']
        if pred == 'Tumor':
            overlay = np.zeros((*preprocessed.shape, 4))
            overlay[:, :, 0] = 1.0  # Red
            overlay[:, :, 3] = 0.3
            axes[1].imshow(overlay)
            axes[1].set_title(f'⚠️ TUMOR ({result["confidence"]:.1f}%)', 
                            color='#e74c3c', fontweight='bold')
        elif pred == 'Liver':
            overlay = np.zeros((*preprocessed.shape, 4))
            overlay[:, :, 2] = 1.0  # Blue
            overlay[:, :, 3] = 0.2
            axes[1].imshow(overlay)
            axes[1].set_title(f'Liver ({result["confidence"]:.1f}%)', 
                            color='#3498db', fontweight='bold')
        else:
            axes[1].set_title(f'Background ({result["confidence"]:.1f}%)', 
                            color='#95a5a6', fontweight='bold')
        axes[1].axis('off')
        
        for ax in axes:
            ax.set_facecolor('#1a1a2e')
        
        plt.tight_layout()
        
        # Embed in tkinter
        canvas = FigureCanvasTkAgg(fig, master=self.fig_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def run(self):
        """Start the application."""
        print("🏥 Starting Liver Cancer Detection App...")
        self.root.mainloop()


def main():
    app = LiverCancerApp()
    app.run()


if __name__ == '__main__':
    main()
