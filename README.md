<div align="center">
<h1 align="center">
<img src="https://raw.githubusercontent.com/PKief/vscode-material-icon-theme/ec559a9f6bfd399b82bb44393651661b08aaf7ba/icons/folder-markdown-open.svg" width="100" />
<br>face-detection
</h1>
<h3>◦ Developed with the software and tools listed below.</h3>

<p align="center">
<img src="https://img.shields.io/badge/Python-3776AB.svg?style&logo=Python&logoColor=white" alt="Python" />
<img src="https://img.shields.io/badge/Markdown-000000.svg?style&logo=Markdown&logoColor=white" alt="Markdown" />
<img src="https://img.shields.io/badge/JSON-000000.svg?style&logo=JSON&logoColor=white" alt="JSON" />
</p>
</div>

---

## 📒 Table of Contents
- [📒 Table of Contents](#-table-of-contents)
- [📍 Overview](#-overview)
- [🧩 Modules](#modules)
- [🚀 Getting Started](#-getting-started)
- [📄 License](#-license)
---


## 📍 Overview

This program has the main feature of classifying an image to a face and non-face image classification. The model that is being currently used is a CNN model.

---



## 🧩 Modules

<details closed><summary>Root</summary>

| File                                                                                          | Summary                   |
| ---                                                                                           | ---                       |
| [config_utils.py](https://github.com/diazoangga/face-detection/blob/main/config_utils.py) | All the setting customization is stored here for dataset generation, model settings, training settings, and output folder naming |
| [dataset_gen.py](https://github.com/diazoangga/face-detection.git/blob/main/dataset_gen.py)   | Dataset generation |
| [evaluate.py](https://github.com/diazoangga/face-detection.git/blob/main/evaluate.py)         | Evaluate the training results |
| [inference.py](https://github.com/diazoangga/face-detection.git/blob/main/inference.py)       | Testing the training results |
| [model.py](https://github.com/diazoangga/face-detection.git/blob/main/model.py)               | MTraining model |
| [training.py](https://github.com/diazoangga/face-detection.git/blob/main/training.py)         | Training code |

</details>
---

## 🚀 Getting Started

### ✔️ Prerequisites

Before you begin, ensure that you have the following prerequisites installed:
> - `ℹ️ tensorflow = 2.9.1`
> - `ℹ️ pyyaml = 6.0`
> - `ℹ️ numpy = 1.21.6`
> - `ℹ️ matplotlib = 3.5.2`
> - `ℹ️ Pillow = 9.1.1`
> - `ℹ️ opencv-contrib-python = 4.5.5.64`

### 📦 Installation

1. Clone the face-detection repository:
```sh
git clone https://github.com/diazoangga/face-detection.git
```

2. Change to the project directory:
```sh
cd face-detection
```

3. Install the dependencies:
```sh
pip install -r requirements.txt
```

### 🎮 Generate Datasets

1. Open "config.yml"
2. Set every parameter below:
>   - RAW_DATA_DIR    : the downloaded dataset path (raw dataset)
>   - POS_DATASET_NUM : number of raw dataset #1237
>   - DATASET_DIR     : path that you want to save you dataset to after being preprocessed
>   - VAL_RATIO       : ratio between validation and training dataset

3. Run "dataset_gen.py" on the command prompt
   ```sh
    python dataset_gen.py
    ```

### 🎮 Train face-detection

1. Open "config.yml"
2. Set every parameter below:
    - TRAIN:
    -   IMG_HEIGHT      : 256
    -   IMG_WIDTH       : 256
    -   MODEL_SAVE_DIR  : './train_out'
    -   TBOARD_SAVE_DIR : './logs/fit'
    -   BATCH_SIZE      : 32
    -   EPOCH_NUMS      : 70
    - SOLVER:
    -   INITIAL_LR      : 0.001
    -   FINAL_LR        : 0.0001

3. Run "dataset_gen.py" on the command prompt
   ```sh
    python training.py
    ```
### 🧪 Evaluate the Trained Model
1. Open "config.yml"
2. Set every parameter below:
    - TEST:
    - WEIGHT_FILE     : weight file for the model, in this case --> './New_folder/train_out/conv256_32_sigmoid/img_class.epoch12-loss0.04.h5'
    - TEST_FILE       : path consisting of the testing dataset
    - EVAL_RESULT_DIR : './eval_out'
    - TEST_BATCH_SIZE : 1
    - THRESHOLD       : Set the threshold value (in this case 0.82)
    - CALCULATE_THRESHOLD: False
3. Run "evaluate.py" and it will calculate the optimum threshold for the model at the end of line.
   ``` sh
   python evaluate.py
   ```
5. optional: it can calculate the optimum threshold value by enabling "CALCULATE_THRESHOLD" into "True".
6. After running "python evaluate.py", you can see the graph of threshold-accuracy on './eval_out/Threshold-Accuracy_Graph.png'. And you can change the "THRESHOLD" on "config.yml" to the optimum threshold value

### 🧪 Inference

1. Open "config.yml"
2. There are 2 ways to do the inference phase:
    - MODE: 'IMAGE'
    - MODE: 'DATASET'
    The first one is to classify face-nonface with the input of a single image.
    The second one is to classify face-nonface with the inputs of predefined testing datasets and output the result in a JSON file

3. If you want to classify face-nonface with a predetermined dataset and generate the JSON file consisting inference-result:
    Change "MODE" in "config.yml" to "DATASET" and change the "TEST_FILE" in the "config.yml" to a path that consists of the dataset.
    and run "python inference.py". The file will be saved as './eval_out_test_dataset_result.json'

4. If you want to run the classifier of face-nonface with a single input image:
    Change "MODE" in "config.yml" to "IMAGE"
    and run "python inference.py". The result will be shown on the command prompt screen

### NOTES:
1. train-results such as trained weight, tensorboard, model architecture, loss and accuracy will be saved in './train_out'
2. train-log (tensorboard) will be saved in ./logs/fit.
    To access the tensorboard result, go to the command prompt screen and type "tensorboard --logdir logs/fit"
    and go to this URL: http://localhost:6006/
3. the JSON file consisting of the inference result of the given test dataset example is saved on './eval_out/test_dataset_result.json'
4. The threshold that being used is 0.82
5. The weight that is being used is ./train_out/img_class.epoch12-loss0.04.h5'

---

## 🤝 Contributing

Contributions are always welcome! Please follow these steps:
1. Fork the project repository. This creates a copy of the project on your account that you can modify without affecting the original project.
2. Clone the forked repository to your local machine using a Git client like Git or GitHub Desktop.
3. Create a new branch with a descriptive name (e.g., `new-feature-branch` or `bugfix-issue-123`).
```sh
git checkout -b new-feature-branch
```
4. Make changes to the project's codebase.
5. Commit your changes to your local branch with a clear commit message that explains the changes you've made.
```sh
git commit -m 'Implemented new feature.'
```
6. Push your changes to your forked repository on GitHub using the following command
```sh
git push origin new-feature-branch
```
7. Create a new pull request to the original project repository. In the pull request, describe the changes you've made and why they're necessary.
The project maintainers will review your changes and provide feedback or merge them into the main branch.

---

## 📄 License

This project is licensed under the `ℹ️  INSERT-LICENSE-TYPE` License. See the [LICENSE](https://docs.github.com/en/communities/setting-up-your-project-for-healthy-contributions/adding-a-license-to-a-repository) file for additional info.

---
