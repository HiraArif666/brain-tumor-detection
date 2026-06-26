# 🧠 Brain Tumor Detection from MRI Scans

A computer vision deep learning model that classifies brain MRI scans into 4 categories using transfer learning, deployed as an interactive web app.

## ✨ Features
- Upload any brain MRI scan image for instant classification
- Classifies into 4 categories: **Glioma**, **Meningioma**, **Pituitary**, or **No Tumor**
- Shows confidence score and full probability breakdown for each class
- Clean, dark-themed Streamlit interface
- Built using transfer learning on a pretrained ResNet18 backbone

## 🛠️ Tech Stack
| Tool | Purpose |
|---|---|
| PyTorch | Deep learning framework |
| ResNet18 (transfer learning) | Pretrained CNN backbone, fine-tuned for classification |
| Streamlit | Web UI |
| Pillow / OpenCV | Image preprocessing |

## 📊 Model Details
- **Architecture:** ResNet18 pretrained on ImageNet, with a custom classification head
- **Training data:** 5,600 labeled MRI images (Training set)
- **Test accuracy:** **87%**
- **Classes:** Glioma, Meningioma, Pituitary, No Tumor
- **Dataset:** [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) (Kaggle)

## ⚙️ Run Locally

1. Clone the repo
   ```bash
   git clone https://github.com/HiraArif666/brain-tumor-detection.git
   cd brain-tumor-detection
   ```

2. Create virtual environment
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```

3. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

4. Download the dataset from Kaggle and place it in a `dataset/` folder with this structure:
   ```
   dataset/
   ├── Training/
   │   ├── glioma/
   │   ├── meningioma/
   │   ├── notumor/
   │   └── pituitary/
   └── Testing/
       ├── glioma/
       ├── meningioma/
       ├── notumor/
       └── pituitary/
   ```

5. Train the model (or skip if you already have `brain_tumor_model.pth`)
   ```bash
   python train.py
   ```

6. Run the app
   ```bash
   streamlit run app.py
   ```

## 📸 How to Use
1. Upload an MRI scan image (JPG/PNG)
2. Click **Analyze**
3. View the predicted tumor type with confidence score
4. Review the full probability breakdown across all 4 classes

## ⚠️ Note
- The dataset and trained model file are **not included** in this repository due to size constraints — see the steps above to download/train them yourself
- This tool is for educational and portfolio purposes only. It is **not a medical diagnostic tool** and should never be used as a substitute for professional medical evaluation
