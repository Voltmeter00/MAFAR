# MAFAR
MAFAR Dataset is a sound event dataset labelled by 10 independent individuals, containing 10 sets of label results.
All the data and labels can be downloaded in: https://drive.google.com/file/d/1bNffdNqpWpDMlu7SDFOzwHo2IUCzF9r6/view?usp=share_link
Here's an English README for your GitHub repository:

---

This repository contains the necessary files and scripts to replicate the results of the paper "Exploring Differences between Human Perception and Model Inference in Audio Event Recognition".(https://arxiv.org/abs/2409.06580)

## Instructions

### 1. Extract the Dataset

First, extract the `human_label_and_model_inference.zip` file. After extraction, you will find two main folders:

- `label`: Contains the human-labeled sound event data. These labels were annotated by 10 experts and aligned to 86 classes of AudioSet by GPT-4.
- `predictions_10s`: Contains the inference results from six AudioSet pre-trained models applied to the same audio segments.

### 2. Running the Scripts

Follow the steps below to replicate the results:

1. **Run the label reader script**  
   This script reads the human-labeled data:
   ```bash
   python 1_read_all_label.py
   ```

2. **Run the label alignment script**  
   This script aligns the labels using the alignment technique:
   ```bash
   python 2_label_alignment.py
   ```

3. **Run the multi-class analysis script**  
   This script performs multi-class analysis on the aligned labels and model predictions:
   ```bash
   python 3_multi_class_analysis.py
   ```

Once you've completed these steps, you will have successfully reproduced the results.

## Requirements

- Python 3.x
- numpy
- sklearn
---
