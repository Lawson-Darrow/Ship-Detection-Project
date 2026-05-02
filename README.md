# Ship Detection in Satellite Imagery

This repository contains a machine learning project for binary ship detection in
satellite image chips. The project compares traditional machine learning models
trained on hand-crafted image features against a convolutional neural network
trained on raw image pixels.

The main deliverable is a single Colab-ready notebook:

- `Ship_Detection_Project_2.ipynb`

## Project Question

Can traditional machine-learning models trained on engineered RGB, HSV, edge,
texture, and shape features compete with a CNN trained directly on raw satellite
image pixels for ship detection?

## Dataset

The project uses the Kaggle Ships in Satellite Imagery dataset, commonly
distributed as `shipsnet.json`.

Expected dataset characteristics:

- 4,000 total image chips
- 80 x 80 RGB images
- 1,000 ship images
- 3,000 no-ship images
- Scene and location metadata used for leakage diagnostics

The raw dataset file is not committed to this repository because
`shipsnet.json` is larger than GitHub's standard file-size limit. To reproduce
the notebook, place the dataset at:

```text
data/shipsnet.json
```

The notebook can also locate the file from Google Drive when run in Colab.

## Repository Contents

```text
.
├── Ship_Detection_Project_2.ipynb      # Complete Colab-ready analysis notebook
├── README.md                           # Project overview and run instructions
├── data/
│   └── README.md                       # Dataset placement instructions
├── Ship_Detection_Proposal.pdf         # Original project proposal
└── Project 2 Requirements.pdf          # Course project requirements
```

Generated outputs, PowerPoint files, temporary files, and raw data are excluded
from version control.

## Notebook Workflow

The notebook is designed to run from top to bottom in Google Colab. It performs:

1. Runtime setup and dependency checks
2. Dataset loading, validation, reshaping, and normalization
3. Stratified train/test split with scene-overlap diagnostics
4. Exploratory data analysis and report-ready visualizations
5. Hand-crafted feature extraction and caching
6. Untuned benchmark model training
7. Hyperparameter tuning for traditional ML models
8. Unsupervised learning with PCA, K-Means, and t-SNE
9. CNN training with ResNet18 transfer learning
10. Final supervised model comparison
11. Scene-held-out generalization evaluation
12. Artifact index and zip packaging for report assets

## Models

The supervised models include:

- Logistic Regression baseline
- Untuned Random Forest baseline
- Tuned Random Forest
- Tuned RBF SVM
- Tuned XGBoost, with a scikit-learn fallback if XGBoost is unavailable
- PCA-reduced Logistic Regression extension
- ResNet18 transfer-learning CNN, with a custom CNN fallback

The unsupervised analysis includes:

- PCA explained variance and 2D projection
- K-Means clustering and label-agreement metrics
- t-SNE feature-space visualization

## Evaluation

Models are evaluated with:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion matrices
- ROC curves

The notebook also compares two evaluation protocols:

- A proposal-aligned stratified chip-level split
- A stricter scene-held-out split that prevents source-scene overlap between
  train and test data

## Current Results

On the original stratified split, the strongest model was the ResNet18 transfer
learning CNN:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| ResNet18 Transfer Learning | 0.9912 | 0.9801 | 0.9850 | 0.9825 | 0.9990 |
| Tuned RBF SVM | 0.9675 | 0.9223 | 0.9500 | 0.9360 | 0.9891 |
| Tuned XGBoost | 0.9662 | 0.9220 | 0.9450 | 0.9333 | 0.9893 |
| Tuned Random Forest | 0.9538 | 0.9179 | 0.8950 | 0.9063 | 0.9872 |
| Logistic Regression Baseline | 0.9338 | 0.8128 | 0.9550 | 0.8782 | 0.9823 |
| PCA-10 Logistic Regression | 0.8388 | 0.6340 | 0.8400 | 0.7226 | 0.9229 |

The scene-held-out evaluation was added because the original stratified split
had source-scene overlap between train and test samples. The grouped evaluation
assigns each source scene entirely to train or test.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Scene-Held-Out ResNet18 Transfer Learning | 0.9988 | 1.0000 | 0.9953 | 0.9977 | 1.0000 |
| Scene-Held-Out Tuned RBF SVM | 0.9714 | 0.9286 | 0.9674 | 0.9476 | 0.9924 |
| Scene-Held-Out Tuned XGBoost | 0.9689 | 0.9204 | 0.9674 | 0.9433 | 0.9959 |
| Scene-Held-Out Logistic Regression | 0.9266 | 0.7955 | 0.9767 | 0.8768 | 0.9892 |

These results should be interpreted alongside the split diagnostics. The
scene-held-out protocol controls source-scene overlap, but the grouped split is
not automatically harder for every random seed.

## Running the Notebook

Recommended environment:

- Google Colab
- GPU runtime for CNN training
- Python 3.10+

Steps:

1. Upload or mount `shipsnet.json` so the notebook can find it at
   `data/shipsnet.json` or in the configured Google Drive paths.
2. Open `Ship_Detection_Project_2.ipynb` in Colab.
3. Use `Runtime > Change runtime type > T4 GPU` for the CNN section.
4. Run all cells from top to bottom.
5. Review the generated tables and figures inline.
6. Use the final artifact-packaging cell to create `ship_detection_outputs.zip`.

The notebook writes report assets to an `outputs/` folder during execution.
Those generated files are intentionally not tracked in git.

## Artifact Outputs

The notebook generates reusable report assets such as:

- Dataset summary tables
- Class-balance plots
- Representative image chips
- Pixel distribution plots
- Scene metadata summaries
- Pipeline diagram
- Feature summary tables
- Model metric tables
- Confusion matrices
- ROC curves
- CNN training curves
- Final model comparison plots
- Scene-held-out protocol comparison tables and figures

## Notes

- The project is an individual course project for MTH/CSE 4224 Intro to Machine
  Learning.
- ShipRSImageNet was reviewed as a possible additional dataset, but it is an
  object-detection dataset with bounding-box annotations and is outside the
  binary image-chip classification scope of this project.
- Presentation files are not required for this repository and are ignored.
