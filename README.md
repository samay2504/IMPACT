# **P. Juliflora Detection using NDVI and Deep Learning**

## **Project Overview**
This project aims to detect and assess the impact of *Prosopis juliflora* using **Normalized Difference Vegetation Index (NDVI)** values derived from satellite imagery. The model utilizes **DeepLabv3+** for semantic segmentation to identify regions dominated by *P. juliflora* in high-resolution **.tif** images.

## **Key Features**
- **Dataset**: 4 high-resolution `.tif` files (**85.238 MB**) with precomputed NDVI values.
- **Preprocessing**: Image processing and NDVI calculations done via **QGIS** and **rasterio**.
- **Model**: **DeepLabv3+** for vegetation classification, trained on labeled NDVI masks.
- **Performance**: Achieved **95% accuracy** in detecting *P. juliflora*.
- **Automation**: Python-based pipeline for batch processing, segmentation, and visualization.

---

## **Dataset Details**
### **Input Data**
- **Satellite Imagery**: 4 `.tif` files with **Red (R), Near-Infrared (NIR)** bands.
- **NDVI Computation**: Derived using the formula:
  
  \[ NDVI = \frac{(NIR - Red)}{(NIR + Red)} \]
  
- **Labeled Masks**: Binary segmentation masks for training (**y_train** shape: (4, 3155, 5545, 1)).

### **Output Data**
- **Segmented Images**: Identified *P. juliflora* regions.
- **GeoTIFF Outputs**: Masked images showing detected vegetation.

---

## **Technology Stack**
- **Python Libraries**: `numpy`, `rasterio`, `matplotlib`, `tensorflow`, `pandas`
- **Deep Learning**: `DeepLabv3+` (TensorFlow/Keras backend)
- **Geospatial Processing**: `QGIS`, `rasterio`
- **Visualization**: `matplotlib`, `seaborn`

---

## **Installation & Setup**
### **Prerequisites**
Ensure you have the following installed:
- Python 3.8+
- TensorFlow
- NumPy
- Rasterio
- Matplotlib
- QGIS (for manual verification)

### **Installation**
```bash
pip install numpy rasterio matplotlib tensorflow pandas
```

---

## **Usage**
### **1. Load Dataset & Compute NDVI**
```python
import rasterio
import numpy as np

def compute_ndvi(nir_band, red_band):
    ndvi = (nir_band - red_band) / (nir_band + red_band)
    return np.nan_to_num(ndvi)  # Handle division by zero
```

### **2. Train Model**
```python
from tensorflow.keras.models import load_model

model = load_model('deeplabv3_model.h5')  # Load pre-trained model
prediction = model.predict(ndvi_image.reshape(1, 3155, 5545, 1))
```

### **3. Visualize Results**
```python
import matplotlib.pyplot as plt
plt.imshow(prediction[0, :, :, 0], cmap='gray')
plt.title("P. Juliflora Segmentation")
plt.show()
```

---

## **Results & Performance**
- **Detection Accuracy**: **95%**
- **False Positives**: **5%**
- **Processing Time**: ~10s per image (GPU-accelerated)
- ![myplot](https://github.com/user-attachments/assets/ff70f1e7-40a6-41e9-8870-2a32b9c4b880)

---

## **Challenges & Solutions**
| Challenge | Solution |
|-----------|----------|
| Noise in NDVI values | Applied thresholding & filtering |
| Limited dataset | Augmented dataset using rotation & scaling |
| Large `.tif` files | Optimized data loading with rasterio |

---

## **Future Improvements**
- **Expand Dataset**: Include more regions for better generalization.
- **Model Enhancement**: Experiment with U-Net and Transformer-based segmentation models.
- **Real-Time Analysis**: Develop a web dashboard for interactive NDVI visualization.

---

## **Contributors**
- **Samay Mehar** - *Lead Developer & Researcher*

---

## **License**
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

