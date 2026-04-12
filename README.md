# Lung Nodule Malignancy Classification (NoduleMNIST3D)

This project trains a small 3D CNN to classify lung nodules as benign vs malignant from 3D CT crops using the MedMNIST NoduleMNIST3D dataset.

- [Lung Nodule Malignancy Classification (NoduleMNIST3D)](#lung-nodule-malignancy-classification-nodulemnist3d)
  - [Project Env Setup](#project-env-setup)
  - [Links to Files](#links-to-files)

---
## Project Env Setup 

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m ipykernel install --user --name lung-nodule-3d
```


## Links to Files
- [Jupyter Notebook](https://github.com/eremozdemir/lung-nodule-3d/blob/develop/notebooks/01_nodulemnist3d.ipynb)
- [Model trial results](https://github.com/eremozdemir/lung-nodule-3d/tree/develop/results/runs)
- [3D CNN Model](https://github.com/eremozdemir/lung-nodule-3d/blob/develop/src/model3d.py)
- [Model Metrics](https://github.com/eremozdemir/lung-nodule-3d/blob/develop/src/metrics.py)
- [Implementaion Notes](https://github.com/eremozdemir/lung-nodule-3d/blob/develop/Implementation_notes.md)



