# MAFAR
MAFAR Dataset is a sound event dataset labelled by 10 professional annotators, containing 10 sets of label results.
All the data and labels can be downloaded in: https://drive.google.com/file/d/1bNffdNqpWpDMlu7SDFOzwHo2IUCzF9r6/view?usp=share_link

This repository contains the necessary files and scripts to replicate the results of the paper **"Exploring Differences between Human Perception and Model Inference in Audio Event Recognition"**.(https://arxiv.org/abs/2409.06580)

## Dataset Contents
- **raw_data**: Unprocessed data labeled by 10 professional annotators.
- **raw_audio**: The original audio of the corresponding clip.
- **result**: Contains the results of individual annotations.
- **foreground**: Annotators' label of what they consider to be foreground events of the corresponding clip.
- **tagged_data**: Labels are processed and correspond to the Audioset categorization.
## Annotation Format
- **annotations/**: Contains `.txt` files with the following columns:
  - `Name_of_audio_segment `: Name of the labeled audio clip. The last two digits represent the number of segments in a audio clip.
  - `start_time`: The start time of the event in the audio file (in seconds).
  - `end_time`: The end time of the event in the audio file (in seconds).
  - `semantic_label`: Semantic meaning related to the event.
  - `Audioset_label`: The transformation of `semantic_label`.

**Annotation Format of raw data**
Each annotation file is a TXT file structured as follows:

| Name_of_audio_segment            | start_time | end_time      | semantic_label   |
|----------------------------------|------------|---------------|------------------|
| DJI_20231202102104_0001_D_01.wav | 0.08       | 2.36          | 两个人的说话声      |
| DJI_20231202102104_0001_D_02.wav | 25.67      | 33.84         | 广播员播报的声音    |
| DJI_20231202102104_0001_D_03.wav | 34.17      | 37.65         | 人说话的声音       |

**Annotation Format of tagged data**
Each annotation file is a TXT file structured as follows:
| Name_of_audio_segment            | start_time | end_time      | Audioset_label      |
|----------------------------------|------------|---------------|---------------------|
| DJI_20231202102104_0001_D_01.wav | 0.08       | 2.36          | Conversation        |
| DJI_20231202102104_0001_D_02.wav | 25.67      | 33.84         | Narration, monologue|
| DJI_20231202102104_0001_D_03.wav | 34.17      | 37.65         | Speech              |



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
