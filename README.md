# Resumaverick

Resumaverick is a BERT-based resume classification tool that fine-tunes a pre-trained Transformer model to categorize resumes into predefined categories. It supports data augmentation techniques, evaluation metrics, and includes utility scripts for training, GPU monitoring, and environment setup.
## Table of Contents
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Training](#training)
- [Data](#data)
- [Testing](#testing)
- [Contributing](#contributing)
- [Author](#author)
#
## Features
- Fine-tune `microsoft/deberta-v3-base` (GPU) or `distilbert-base-uncased` (CPU) for resume classification
- Data augmentation: synonym replacement, back-translation, sentence shuffling
- Evaluation metrics: accuracy and weighted F1 score
- Checkpointing and best-model selection during training
- Utility scripts for environment setup, GPU monitoring, and parallel processing examples
#
## Project Structure
```
.
├── data/                         # Resume dataset (Resume.csv)
├── models/                       # Saved/final models
│   └── bert-classifier/
├── results/                      # Training checkpoints and logs
├── resumaverick/                 # Package source code
│   ├── augmentation.py           # Data augmentation functions
│   ├── bert_classifier.py        # Training & evaluation script
│   └── utils.py                  # Data loading utilities
├── tests/                        # Unit tests (augmentation)
├── run.py                        # Parallel processing example (uses tqdm.contrib.concurrent)
├── train_bert.sh                 # Shortcut to train BERT model
├── download_language_data.sh     # Download SpaCy language models
├── install_debian.sh             # Debian setup (Python 3.12, Poetry, deps)
├── monitor_gpu.sh                # GPU monitoring helper
└── spin_spot_gpu_gcp.sh          # GCP spot instance helper
```
#
## Installation
1. Clone the repository:
   ```bash
   git clone <repo_url>
   cd resumaverick
   ```
2. (Optional) On Debian/Ubuntu systems, run the installer script:
   ```bash
   sh install_debian.sh
   ```
   This installs Python 3.12, Poetry, dependencies, and SpaCy models.
3. Alternatively, install with Poetry manually:
   ```bash
   poetry install
   sh download_language_data.sh
   ```
4. Ensure you have `Resume.csv` in the `data/` directory.
#
## Usage
### Training
To fine-tune the BERT classifier on your resume dataset:
```bash
sh train_bert.sh
```
- Trained models and best checkpoints are saved under `models/bert-classifier` and `results/`.

## Data
- Based on Kaggle dataset: https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset
- Alternatively place your resume data as a CSV in `data/Resume.csv` with at least the following columns:
  - `Resume_str`: raw text of the resume
  - `Category`: label/category for classification
#
## Testing
Run the unit tests for data augmentation:
```bash
pytest tests
```  
#
## Contributing
Contributions are welcome! Please open issues or pull requests to suggest improvements.
#
## Author
GrossBetruger 
