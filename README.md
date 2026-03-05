# Lung Nodule Malignancy Classification (NoduleMNIST3D)

This project trains a small 3D CNN to classify lung nodules as benign vs malignant from 3D CT crops using the MedMNIST NoduleMNIST3D dataset.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m ipykernel install --user --name lung-nodule-3d