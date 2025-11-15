# Medical Image Regression - Student Project

## Project Overview

This is an educational project for medical image regression tasks. The objective is to predict disease scores (score_avg) based on X-ray images.

### Dataset Information

- **Training Set**: 500 images (sub_train/)
- **Validation Set**: 100 images (sub_val/)
- **CSV File**: data.csv contains image metadata and labels

### Evaluation Metrics

- **MAE** (Mean Absolute Error): Average absolute error
- **RMSE** (Root Mean Square Error): Root mean square error
- **R²** (R-squared): Coefficient of determination

---

## Project Structure

```
.
├── data.csv              # Dataset CSV file
├── sub_train/            # Training images directory
├── sub_val/              # Validation images directory
├── dataset.py            # Dataset loading module
├── model.py              # Model definition module
├── utils.py              # Utility functions module
├── main.py               # Main training script
├── inference.py          # Inference script
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
```

---

## Environment Setup

### 1. Create Virtual Environment (Recommended)

```bash
# Using conda
conda create -n medical_img python=3.8
conda activate medical_img

# Or using venv
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Quick Start

### 1. Train Baseline Model (SimpleCNN)

```bash
python main.py --model simple_cnn --epochs 50 --batch_size 32 --lr 0.001
```

### 2. Inference on Single Image

```bash
python inference.py --checkpoint checkpoints/best_model.pth --image_path sub_val/DX100001.jpg --model simple_cnn
```

### 3. Batch Inference

```bash
python inference.py --checkpoint checkpoints/best_model.pth --image_dir sub_val --model simple_cnn --output results.csv
```

---

## Command Line Arguments

### main.py Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--csv_file` | str | data.csv | Path to CSV file |
| `--train_dir` | str | sub_train | Training images directory |
| `--val_dir` | str | sub_val | Validation images directory |
| `--model` | str | simple_cnn | Model type (simple_cnn/student) |
| `--epochs` | int | 50 | Number of training epochs |
| `--batch_size` | int | 32 | Batch size |
| `--lr` | float | 0.001 | Learning rate |
| `--weight_decay` | float | 1e-4 | Weight decay |
| `--image_size` | int | 224 | Image size |
| `--optimizer` | str | adam | Optimizer (adam/sgd/adamw) |
| `--scheduler` | str | plateau | Learning rate scheduler |
| `--early_stopping` | int | 15 | Early stopping patience |
| `--save_dir` | str | checkpoints | Checkpoint save directory |
| `--num_workers` | int | 4 | Number of data loading workers |
| `--seed` | int | 42 | Random seed |

---

## Student Tasks

### Task Requirements

1. **Understand Code Structure** (20%)
   - Read and understand dataset.py, model.py, utils.py, main.py
   - Understand the training process and evaluation methods

2. **Run Baseline Model** (20%)
   - Train SimpleCNN model
   - Record training process and final performance
   - Analyze training curves and prediction results

3. **Implement Your Own Model** (40%)
   - Implement your model in the `StudentModel` class in model.py
   - You can consider the following approaches:
     - Modify SimpleCNN structure (deeper/wider)
     - Use pretrained models (ResNet, EfficientNet, etc.)
     - Add attention mechanisms
     - Use residual connections
     - Try different regularization methods

4. **Hyperparameter Tuning** (10%)
   - Try different learning rates, batch sizes, optimizers, etc.
   - Use different data augmentation strategies
   - Record experiment results

5. **Write Report** (10%)
   - Model architecture description
   - Experimental setup and results
   - Performance comparison and analysis
   - Improvement ideas and conclusions

### Grading Criteria

- **Code Quality**: Clear code with complete comments
- **Model Performance**: Performance on validation set (MAE, RMSE, R²)
- **Innovation**: Degree of innovation in model design
- **Report Quality**: Depth and completeness of experimental analysis

### Tips

1. **Start Simple**: Run SimpleCNN first to understand the workflow
2. **Incremental Improvement**: Modify one component at a time and observe effects
3. **Record Experiments**: Use an experiment log table to record all attempts
4. **Visualize Analysis**: Use training curves and prediction plots for analysis
5. **Reference Literature**: Review related papers on medical image analysis

---

## Model Description

### SimpleCNN (Baseline Model)

Simple 4-layer convolutional neural network, including:
- 4 convolutional blocks (Conv2d + BatchNorm + ReLU + MaxPool)
- 2 fully connected layers
- Dropout regularization

**Expected Performance**: MAE ~20-30

### StudentModel (To Be Implemented)

Model to be designed and implemented by students.

---

## Experiment Log Template

| Exp ID | Model | LR | Batch Size | Epochs | MAE | RMSE | R² | Notes |
|--------|-------|-----|-----------|--------|-----|------|----|-------|
| 1 | SimpleCNN | 0.001 | 32 | 50 | - | - | - | Baseline |
| 2 | Student | ... | ... | ... | - | - | - | Custom |

---

## Common Issues

### Q1: Training is slow?
- Reduce `batch_size`
- Reduce `num_workers`
- Use smaller `image_size`
- If GPU is available, ensure CUDA is enabled

### Q2: Out of memory?
- Reduce `batch_size`
- Reduce `image_size`
- Close other programs

### Q3: Model overfitting?
- Increase data augmentation
- Increase Dropout
- Reduce model complexity
- Increase weight decay

### Q4: Model underfitting?
- Increase model complexity
- Increase training epochs
- Adjust learning rate
- Reduce regularization

---

## Output Files

After training completes, files will be generated in `checkpoints/{model_name}_{timestamp}/`:

- `best_model.pth`: Best model on validation set
- `latest_model.pth`: Model from last epoch
- `training_history.png`: Training curves plot
- `predictions.png`: Predictions vs true values scatter plot
- `results.txt`: Detailed training configuration and results

---

## Reference Resources

### Deep Learning Frameworks
- [PyTorch Official Documentation](https://pytorch.org/docs/stable/index.html)
- [PyTorch Tutorial](https://pytorch.org/tutorials/)

### Model Architectures
- [CNN Basics](https://cs231n.github.io/)
- [ResNet Paper](https://arxiv.org/abs/1512.03385)
- [EfficientNet Paper](https://arxiv.org/abs/1905.11946)

### Medical Image Analysis
- Related survey papers
- Kaggle medical image competitions

---

## Contact

For questions, please contact:
- Instructor Email: [your_email@example.com]
- Course Website: [course_website]

---

## License

This project is for educational purposes only.

---

## Changelog

- 2025-10-13: Project initialization
  - Created dataset and baseline model
  - Completed training and inference scripts
  - Added documentation
