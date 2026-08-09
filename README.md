# LegalIR

## Step 1: Environment Setup

Assuming Python 3.12 is installed:

1. Create the virtual environment:
```bash
python -m venv dsc_env
```

2. Activate the virtual environment (Git Bash):
```bash
source dsc_env/Scripts/activate
```

3. Install PyTorch matching your machine's CUDA version:
> Refer to [PyTorch Get Started](https://pytorch.org/get-started/locally/) to select the installation command appropriate for your CUDA version or CPU setup.

Example for CUDA 12.6:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

4. Install remaining dependencies:
```bash
pip install -r requirements.txt
```

## Step 2: Dataset Preparation

1. Download the public test dataset from the Drive link: [Drive](https://drive.google.com/drive/folders/1e4XctfiDz9TNPuxYtNJ3Uoaz0vQ9gB1t)
2. Extract the downloaded dataset into the `public_test_dataset/` folder.

Directory structure:
```
public_test_dataset/
├── DSC2026_Task1_LegalIR_Data_Overview.docx
├── public-official.json
├── train.json
└── selected-contexts/
    ├── context_100050.json
    ├── context_100062.json
    └── ...
```
