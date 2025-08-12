# OCT–Biomarker Alignment
OCT–Biomarker alignment performs CLIP-style semantic alignment between RETFound and 33 biomarkers from the UK Biobank.

## Download the weights for RETFound
Please download the OCT-based weights from the RETFound repository: [RETFound_MAE](https://github.com/rmaphoh/RETFound_MAE).

## Orginize the data
Please organize the data as:
- Dataset
  - training dataset
  - test dataset
  - training.csv
  - test.csv

The CSV files should be organized as shown in the figure below.

![Dataset layout](./data_architecture.png)

## Training Model
### 🔧 Install environment
1. Create environment with conda:

```bash
conda create -n Aignment python=3.11.0 -y
conda activate Alignment
```
2: Install dependencies
```bash
conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=12.1 -c pytorch -c nvidia
git clone https://github.com/rmaphoh/RETFound_MAE/
cd RETFound_MAE
pip install -r requirements.txt
```


