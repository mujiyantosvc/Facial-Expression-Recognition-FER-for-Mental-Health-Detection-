# Facial Expression Recognition (FER) for Mental Health Detection Using Transformer Model

![Python 3.10](https://img.shields.io/badge/python-3.10-green.svg?style=plastic)
![PyTorch 2.0](https://img.shields.io/badge/pytorch-2.0-green.svg?style=plastic)
![CUDA 11](https://img.shields.io/badge/cuda-11-green.svg?style=plastic)
![License CC BY 4.0](https://img.shields.io/badge/license-MIT-green.svg?style=plastic)

Welcome to the **Facial Expression Recognition (FER) for Mental Health Detection** repository. This project leverages cutting-edge AI models, including **Swin Transformer**, to analyze facial expressions for detecting mental health conditions. For detailed insights, refer to the [research paper published in Engineering, Technology & Applied Science Research](https://doi.org/10.48084/etasr.9139), indexed in Scopus Q2.

---

## 📘 Overview of Facial Expression Recognition Techniques Using Python

Mental health issues such as **anxiety**, **depression**, **OCD (Obsessive Compulsive Disorder)**, **PTSD (Post-Traumatic Stress Disorder)**, and other conditions significantly impact individuals and society. Early detection and intervention can drastically improve outcomes, and **Facial Expression Recognition (FER)** provides a non-invasive and efficient way to monitor emotional states.

This repository combines **Artificial Intelligence for Mental Health** with advanced **Facial Emotion Recognition** techniques to identify subtle changes in expressions that indicate mental health risks. The project leverages cutting-edge models, including **Swin Transformers**, **Vision Transformers (ViT)**, and **Custom CNNs**, integrated with robust datasets such as **FER2013** and **CK+**. These models are designed to:

- Recognize emotions like happiness, sadness, anger, fear, and surprise.
- Detect early signs of mental health conditions such as **serious mental illness** and stress-related disorders.
- Provide practical applications in **AI Emotion Recognition** for healthcare, HR, and research.

**Key Features:**

- High-accuracy emotion detection using **deep learning for facial expression recognition**.
- Integration with **mental health scoring systems** to quantify emotional health.
- Applications in **real-time emotion detection systems** and **emotion detection using OpenCV Python**.

---

## 📂 AI for Mental Health Repository Structure

```
📦FER-for-Mental-Health-Detection
├── 📁 Models
│   ├── 📁 Swin_Transformer
│   ├── 📁 Custom_CNN
│   ├── 📁 ViT_Model
│   └── 📁 Other_Models
├── 📁 datasets
├── 📁 images
├── 📁 utilities
├── 📄 README.md
├── 📄 usage_guide.md
├── 📄 LICENSE
└── 📄 requirements.txt
```

- **Models**: Contains different subfolders for Swin Transformer, Custom CNN, ViT, and other models.
- **datasets**: Includes FER2013, CK+, and Genius HR datasets.
- **images**: Visualizations such as Grad-CAM, architecture diagrams, and augmented samples.
- **utilities**: Scripts for data preprocessing, augmentation, and evaluation.
- **usage_guide.md**: Detailed guide to using the models.

---

## 📚 Datasets

### FER2013

- **Description**: A dataset of 35,887 grayscale images labeled with seven emotions (Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral).
- **Source**: [FER2013 on Kaggle](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge).

### CK+

- **Description**: A smaller dataset of 920 images with eight emotion labels.
- **Source**: [CK+ Dataset Official Site](https://www.jeffcohn.net/Resources/).

### Genius HR Dataset

- **Description**: A real-world dataset for workplace mental health analysis.
- **Source**: Proprietary dataset. Contact for access.

---

## 🚀 Emotion Detection Using Python Installation

### Prerequisites

- Python 3.10+
- PyTorch 2.0+
- CUDA-enabled GPU (recommended)

### Installation Steps

Follow these detailed steps to set up the project and avoid any errors during installation.

---

#### 1. Clone the Repository

Start by cloning the repository to your local machine and navigate into the project directory:

```bash
# Clone the repository
git clone https://github.com/mujiyantosvc/Facial-Expression-Recognition-FER-for-Mental-Health-Detection-.git

# Navigate into the project directory
cd Facial-Expression-Recognition-FER-for-Mental-Health-Detection-
```

---

#### 2. Create a Virtual Environment

It is highly recommended to use a virtual environment to manage dependencies. Run the following commands based on your operating system:

```bash
# For Linux/MacOS
python3 -m venv venv
source venv/bin/activate

# For Windows
python -m venv venv
.\venv\Scripts\activate
```

---

#### 3. Update pip and Install Dependencies

Ensure your `pip` is up-to-date and install all required dependencies listed in `requirements.txt`:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

#### 4. Verify Installation

To confirm that everything is installed correctly, run the following commands:

```bash
# Check Python version
python --version
# Output should be Python 3.10+

# Check PyTorch installation
python -c "import torch; print(torch.__version__)"
# Output should match the PyTorch version specified in requirements.txt
```

---

#### 5. Download the FER2013 Dataset

This project uses the FER2013 dataset. Follow these steps to download and prepare the dataset:

```bash
# Install Kaggle CLI
pip install kaggle

# Move your Kaggle API token to the appropriate location
mkdir ~/.kaggle
mv /path/to/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# Download the FER2013 dataset
kaggle competitions download -c challenges-in-representation-learning-facial-expression-recognition-challenge

# Extract the dataset into the datasets/ folder
mkdir datasets
unzip challenges-in-representation-learning-facial-expression-recognition-challenge.zip -d datasets
```

---

#### 6. Preprocess the Dataset

Organize the dataset into train, validation, and test directories:

```bash
python utilities/preprocess_data.py
```

---

#### 7. Run the Model

To ensure everything is working, run the default Swin Transformer model training script:

```bash
python utilities/train_model.py --model swin_transformer --epochs 10 --batch_size 32
```

This command trains the Swin Transformer model on the FER2013 dataset with default settings.

---

### Troubleshooting

If you encounter issues during installation:

- **CUDA errors**: Ensure you have the correct version of CUDA installed.
- **Dependency conflicts**: Manually resolve versions in `requirements.txt`.
- **Dataset errors**: Verify that the FER2013 dataset is correctly downloaded and extracted.

---

### Installation Complete!

You are now ready to explore Facial Expression Recognition for Mental Health Detection. For more information, refer to the [Usage Guide](./usage_guide.md).

---

## 💡 Models and Architectures

### 1. Swin Transformer

- **Description**: A hierarchical transformer optimized for visual tasks, ideal for **facial expression recognition** and mental health detection.
- **Reference**: [Swin Transformer Paper](https://arxiv.org/abs/2103.14030)

### 2. Custom CNN

- **Description**: Lightweight CNN for real-time emotion detection, suitable for **AI Emotion Recognition** tasks.

### 3. Vision Transformer (ViT)

- **Description**: Captures long-range dependencies in facial features for robust **facial emotion recognition**.
- **Reference**: [ViT Paper](https://arxiv.org/abs/2010.11929)

### 4. Additional Models

- Includes MobileNet, EfficientNet, and hybrid architectures for **real-time emotion detection**.

---

## 📷 Visualizations

### Augmented Images

![Augmented Images](./images/facial-expression-recognition-augmented-dataset.jpg)

- Visualizes data augmentation techniques used to enhance model robustness.

### Model Architecture

![FER Architecture](./images/facial-expression-recognition-swin-transformer-model-architecture.jpg)

- Diagram of the Swin Transformer model optimized for **facial expression recognition**.

### Grad-CAM Visualizations

![Grad-CAM Visualization](./images/facial-emotion-recognition-grad-cam-visualizations.jpg)

- Highlights the facial regions influencing the model’s predictions.

### Mental Health Scoring Summary

| **Employee ID** | **Avg Confidence** | **No. of Images** | **Mental Health Score** |
| --------------- | ------------------ | ----------------- | ----------------------- |
| 31              | 0.7747             | 30                | 52.03                   |
| 39              | 0.9230             | 30                | 53.00                   |
| 16              | 0.8943             | 30                | 53.00                   |
| 15              | 0.6484             | 30                | 50.93                   |
| 17              | 0.7503             | 30                | 51.07                   |

---

## 📈 Applications

- **Human Resources**: Monitor and assess employee mental health using **AI for mental health detection**.
- **Healthcare**: Real-time emotion detection for early mental health interventions.
- **Research**: Advance the field of **artificial intelligence in mental health detection**.

---

## 📄 Citation

This research has been published in **Engineering, Technology & Applied Science Research**, indexed in **Scopus Q2**. Below is the certification evidence:

<img src="./images/scopus-fer.jpg" alt="Scopus Q2 Certification" width="200">

### Citation Formats

**APA:**

> Mujiyanto, M., Setyanto, A., Kusrini, K., & Utami, E. (2024). Swin Transformer with Enhanced Dropout and Layer-wise Unfreezing for Facial Expression Recognition in Mental Health Detection. Engineering, Technology & Applied Science Research, 14(6), 19016–19023. https://doi.org/10.48084/etasr.9139

**MLA:**

> Mujiyanto, M., et al. "Facial Expression Recognition (FER) for Mental Health Detection." Engineering, Technology & Applied Science Research, vol. 14, no. 6, 2024, pp. 19016-19023.

**Vancouver:**

> Mujiyanto M, et al. Facial Expression Recognition (FER) for Mental Health Detection. Engineering, Technology & Applied Science Research. 2024;14(6):19016-23.

---

## 📧 Contact

For questions or support, please contact:

- **Email**: [mujiyanto@amikom.ac.id](mailto:mujiyanto@amikom.ac.id)

### Special Credit

- **Description**: Indonesia  artificial intelligence AI Developer | Website Developer | Mobile Developer | Software Developer | Software House Indonesia
- **Reference**: [Second Vision Corp](https://secondvisioncorp.com/)
