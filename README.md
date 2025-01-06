# Facial Expression Recognition (FER) for Mental Health Detection

Welcome to the **Facial Expression Recognition (FER) for Mental Health Detection** repository. This repository provides a comprehensive suite of AI models and tools to analyze facial expressions for detecting mental health conditions such as anxiety, depression, OCD, PTSD, and more. By leveraging advanced deep learning techniques, this project aims to bridge the gap between early mental health detection and effective intervention.

---

## 📘 Overview

Facial Expression Recognition (FER) plays a crucial role in understanding human emotions and detecting early signs of mental health conditions. This repository includes:

- Advanced AI models like Swin Transformers, Vision Transformers (ViT), and Custom CNNs.
- Tools for emotion analysis, such as happiness, sadness, anger, fear, and surprise.
- Real-world applications in healthcare, human resources (HR), and mental health research.

**Key Features:**
- Emotion detection using state-of-the-art AI models.
- Integration with mental health scoring systems.
- Applications in HR, healthcare, and real-time analysis.

---

## 📂 Repository Structure

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
- Description: A dataset of 35,887 grayscale images labeled with seven emotions (Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral).
- Source: [FER2013 on Kaggle](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge).

### CK+
- Description: A smaller dataset of 920 images with eight emotion labels.
- Source: [CK+ Dataset Official Site](https://www.jeffcohn.net/Resources/).

### Genius HR Dataset
- Description: A real-world dataset for workplace mental health analysis.
- Source: Proprietary dataset. Contact for access.

---

## 🚀 Installation

### Prerequisites
- Python 3.10+
- PyTorch 2.0+
- CUDA-enabled GPU (recommended)

### Installation Steps
```bash
# Clone the repository
git clone https://github.com/username/FER-for-Mental-Health-Detection.git

# Navigate to the project directory
cd FER-for-Mental-Health-Detection

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # Linux/MacOS
.\venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## 💡 Models and Architectures

### 1. Swin Transformer
- **Description**: A hierarchical transformer optimized for visual tasks.
- **Reference**: [Swin Transformer Paper](https://arxiv.org/abs/2103.14030)

### 2. Custom CNN
- **Description**: Lightweight CNN for real-time emotion detection.

### 3. Vision Transformer (ViT)
- **Description**: Captures long-range dependencies in facial features.
- **Reference**: [ViT Paper](https://arxiv.org/abs/2010.11929)

### 4. Additional Models
- MobileNet, EfficientNet, and hybrid models will be added in future updates.

---

## 📷 Visualizations

### Augmented Images
![Augmented Images](./images/augmented_images.png)
- Visualizes data augmentation techniques used to enhance model robustness.

### Model Architecture
![FER Architecture](./images/fer_architecture.png)
- Diagram of the Swin Transformer model optimized for FER tasks.

### Grad-CAM Visualizations
![Grad-CAM Visualization](./images/grad_cam_visualization.png)
- Highlights the facial regions influencing the model’s predictions.

### Mental Health Scoring Summary
| **Employee ID** | **Avg Confidence** | **No. of Images** | **Mental Health Score** |
|-----------------|-------------------|-------------------|-------------------------|
| 31              | 0.7747            | 30                | 52.03                   |
| 39              | 0.9230            | 30                | 53.00                   |
| 16              | 0.8943            | 30                | 53.00                   |
| 15              | 0.6484            | 30                | 50.93                   |
| 17              | 0.7503            | 30                | 51.07                   |

---

## 📈 Applications

- **Human Resources**: Monitor and assess employee mental health.
- **Healthcare**: Real-time emotion detection for mental health interventions.
- **Research**: Enhance AI applications in mental health detection.

---

## 🔗 References

- **FER2013 Dataset**: [FER2013 Kaggle](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge)
- **CK+ Dataset**: [CK+ Official Site](https://www.jeffcohn.net/Resources/)
- **Swin Transformer**: [Paper on arXiv](https://arxiv.org/abs/2103.14030)
- **Vision Transformer (ViT)**: [Paper on arXiv](https://arxiv.org/abs/2010.11929)

---

## 📄 Citation

If you use this repository, please cite:

**APA:**
> Mujiyanto, M., et al. (2024). Facial Expression Recognition (FER) for Mental Health Detection. Engineering, Technology & Applied Science Research, 14(6), 19016-19023.

**MLA:**
> Mujiyanto, M., et al. "Facial Expression Recognition (FER) for Mental Health Detection." Engineering, Technology & Applied Science Research, vol. 14, no. 6, 2024, pp. 19016-19023.

**Vancouver:**
> Mujiyanto M, et al. Facial Expression Recognition (FER) for Mental Health Detection. Engineering, Technology & Applied Science Research. 2024;14(6):19016-23.

---

## 📧 Contact

For questions or support, please contact:
- **Email**: [mujiyanto@amikom.ac.id](mailto:mujiyanto@amikom.ac.id)


