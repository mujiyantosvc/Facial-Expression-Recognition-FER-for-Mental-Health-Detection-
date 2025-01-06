# Usage Guide for Facial Expression Recognition (FER) for Mental Health Detection

This guide provides step-by-step instructions to set up, train, and evaluate the models in this repository.

---

## 📦 Dataset Preparation

### Download the FER2013 Dataset

1. **Create a Kaggle API Token**:

   - Login to [Kaggle](https://www.kaggle.com).
   - Go to your profile and click `Create New API Token`.
   - Download the `kaggle.json` file.

2. **Setup Kaggle CLI**:

   ```bash
   mkdir ~/.kaggle
   mv /path/to/kaggle.json ~/.kaggle/
   chmod 600 ~/.kaggle/kaggle.json
   pip install kaggle
   ```

3. **Download the FER2013 Dataset**:

   ```bash
   kaggle competitions download -c challenges-in-representation-learning-facial-expression-recognition-challenge
   ```

4. **Extract the Dataset**:

   ```bash
   mkdir fer2013
   tar -xvzf challenges-in-representation-learning-facial-expression-recognition-challenge.tar.gz -C fer2013
   ```

5. **Preprocess the Dataset**:
   Run the preprocessing script to organize images into train, validation, and test folders:
   ```bash
   python utilities/preprocess_data.py
   ```

---

## 🚀 Training a Model

### Step 1: Select a Model

The repository includes multiple models such as Swin Transformer, Custom CNN, and Vision Transformer. To train a specific model, use the corresponding script in `utilities/train_model.py`.

### Step 2: Train the Model

Run the following command to train the model:

```bash
python utilities/train_model.py --model swin_transformer --epochs 50 --batch_size 64
```

### Parameters:

- `--model`: Specify the model type (`swin_transformer`, `custom_cnn`, etc.).
- `--epochs`: Number of training epochs (default: 50).
- `--batch_size`: Batch size for training (default: 64).
- `--learning_rate`: Learning rate for optimization (default: 1e-4).

The trained model will be saved to the `models/` directory.

---

## 📊 Evaluating a Model

### Step 1: Evaluate the Model

To evaluate a trained model on the test dataset, use:

```bash
python utilities/evaluate_model.py --model swin_transformer --checkpoint models/best_model.pth
```

### Step 2: Output Metrics

The script will generate metrics such as:

- Accuracy
- Precision
- Recall
- F1-Score

It will also display a classification report and confusion matrix.

---

## 🔍 Visualizations

### Grad-CAM Visualization

To visualize the regions influencing the model's predictions, use:

```bash
python utilities/visualize_results.py --model swin_transformer --checkpoint models/best_model.pth
```

This will generate heatmaps highlighting key facial regions for each emotion prediction.

---

## 🛠 Fine-Tuning Options

### Hyperparameters

Adjust hyperparameters in the training script (`train_model.py`):

- `learning_rate`: Controls the step size in gradient descent.
- `weight_decay`: Regularizes the model to prevent overfitting.
- `dropout_rate`: Controls the dropout probability in fully connected layers.

### Layer Freezing/Unfreezing

For transformer-based models, freeze the initial layers during the first 10 epochs:

```python
for name, param in model.backbone.named_parameters():
    if 'layers.0' in name or 'layers.1' in name:
        param.requires_grad = False
```

Unfreeze the layers after the 10th epoch to fine-tune deeper features:

```python
if epoch == 10:
    for name, param in model.backbone.named_parameters():
        param.requires_grad = True
```

---

## 💡 Tips for Best Results

1. **Class Imbalance**:
   Use class weighting to handle imbalanced datasets. Example in `train_model.py`:

   ```python
   class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
   criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights).to(device))
   ```

2. **Data Augmentation**:
   Use aggressive augmentations to improve generalization:

   - Random rotation
   - Color jitter
   - Random erasing

3. **Early Stopping**:
   Monitor validation loss and stop training when no improvement is observed for 10 epochs.

---

## 🌐 References

- **FER2013 Dataset**: [Kaggle Challenge](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge)
- **Swin Transformer**: [Arxiv Paper](https://arxiv.org/abs/2103.14030)
- **Vision Transformer**: [Arxiv Paper](https://arxiv.org/abs/2010.11929)

---

For additional support, refer to the [README.md](./README.md) or contact us at [mujiyanto@amikom.ac.id](mailto:mujiyanto@amikom.ac.id).
