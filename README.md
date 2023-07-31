# Paediatric-MRI-Segmentation-
# Readme for Brats23.ipynb

This notebook is for the BraTS-2023 MICCAI challenge. The International Brain Tumor Segmentation (BraTS) challenge focuses on the generation of a benchmarking environment and dataset for the delineation of adult brain gliomas. BraTS 2023 expands the dataset to ~4,500 cases to address various populations, tumors, clinical concerns, technical considerations, and algorithmic generalizability.

The notebook begins with the `Segmentation - Pediatric Tumors` challenge, which focuses on benchmarking the development of volumetric segmentation algorithms for pediatric brain glioma through standardized quantitative performance evaluation metrics.

## Packages, Libraries, & Configs

### For UNET 2D

The notebook uses various Python packages and libraries for image processing, neural imaging, machine learning, and data handling. Some notable packages include:
- `nilearn` for neural imaging
- `nibabel` for working with NIfTI images
- `monai` for medical image analysis

### For MONAI EXPs

The notebook also uses MONAI (Medical Open Network for AI) experimental functionalities for medical image analysis. MONAI provides various utilities and transforms for preprocessing and augmenting medical image data, making it easier to work with medical imaging tasks.

## Dataset

The dataset used in this notebook is the ASNR-MICCAI-BraTS2023-PED-Challenge-TrainingData and ASNR-MICCAI-BraTS2023-PED-Challenge-ValidationData. The dataset consists of MRI images related to pediatric brain tumors, and it needs to be downloaded from the official BraTS 23 challenge portal.


![brain](https://github.com/ozzmanmuhammad/Paediatric-MRI-Segmentation-/assets/93766242/3aaa484f-bd8e-4df7-a371-a03271191078)

## Example Data Visualization

The notebook displays example data visualization, including various MRI images (T2 Flair, T1-weighted after contrast, T1-weighted post-contrast, and T2-weighted) along with the corresponding segmentation mask. The segmentation mask identifies different classes related to the tumor.

## MONAI Experiments

The notebook proceeds with MONAI experiments for the brain tumor segmentation task. It includes setting up the MONAI transforms, data loader, and using the Swin UNETR model.

### Mapping file.json

The notebook creates a JSON file containing training and validation sets for internal splits. The training data has been split into 90% training and 10% validation sets.

### Setups

The notebook contains setup configurations like `roi` (region of interest), `batch_size`, `sw_batch_size`, `fold`, `infer_overlap`, `max_epochs`, and `val_every`.

### Swin UNETR

The Swin UNETR model architecture is employed for brain tumor semantic segmentation. The Swin UNETR model uses a transformer-based encoder connected to a CNN-decoder via skip connections. The segmentation output consists of 3 channels corresponding to different sub-regions of the tumor. The notebook sets up the Swin UNETR model, optimizer, and loss function.

### Optimizer and Loss Function

The notebook uses the AdamW optimizer and DiceLoss as the loss function for training the model. The DiceLoss measures the overlap between the predicted segmentation and the ground truth.

### Training and Validation Epochs

The notebook contains functions for training and validation epochs. During training, the model is updated using backpropagation and the calculated loss. The model is evaluated during validation using Dice metrics to measure the accuracy.

### Trainer

The trainer function coordinates the training and validation process, updating the model parameters, and saving the best-performing model checkpoint.

### Execute Training

Finally, the notebook executes the training process using the Swin UNETR model, and the performance metrics (Dice scores) are monitored and saved for evaluation.

Note: This is under process so it does contain some uncompleted codes
