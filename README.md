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
<img src="https://img.shields.io/github/languages/top/diazoangga/face-detection.git?style&color=5D6D7E" alt="GitHub top language" />
<img src="https://img.shields.io/github/languages/code-size/diazoangga/face-detection.git?style&color=5D6D7E" alt="GitHub code size in bytes" />
<img src="https://img.shields.io/github/commit-activity/m/diazoangga/face-detection.git?style&color=5D6D7E" alt="GitHub commit activity" />
<img src="https://img.shields.io/github/license/diazoangga/face-detection.git?style&color=5D6D7E" alt="GitHub license" />
</div>

---

## 📒 Table of Contents
- [📒 Table of Contents](#-table-of-contents)
- [📍 Overview](#-overview)
- [⚙️ Features](#-features)
- [📂 Project Structure](#project-structure)
- [🧩 Modules](#modules)
- [🚀 Getting Started](#-getting-started)
- [🗺 Roadmap](#-roadmap)
- [🤝 Contributing](#-contributing)
- [📄 License](#-license)
- [👏 Acknowledgments](#-acknowledgments)

---


## 📍 Overview

HTTPStatus Exception: 429

---

## ⚙️ Features

HTTPStatus Exception: 429

---


## 📂 Project Structure




---

## 🧩 Modules

<details closed><summary>Root</summary>

| File                                                                                          | Summary                   |
| ---                                                                                           | ---                       |
| [config_utils.py](https://github.com/diazoangga/face-detection.git/blob/main/config_utils.py) | HTTPStatus Exception: 429 |
| [dataset_gen.py](https://github.com/diazoangga/face-detection.git/blob/main/dataset_gen.py)   | HTTPStatus Exception: 429 |
| [evaluate.py](https://github.com/diazoangga/face-detection.git/blob/main/evaluate.py)         | HTTPStatus Exception: 429 |
| [inference.py](https://github.com/diazoangga/face-detection.git/blob/main/inference.py)       | HTTPStatus Exception: 429 |
| [model.py](https://github.com/diazoangga/face-detection.git/blob/main/model.py)               | HTTPStatus Exception: 429 |
| [training.py](https://github.com/diazoangga/face-detection.git/blob/main/training.py)         | HTTPStatus Exception: 429 |

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
 
    RAW_DATA_DIR    : the downloaded dataset path (raw dataset)
    POS_DATASET_NUM : number of raw dataset #1237
    DATASET_DIR     : path that you want to save you dataset to after being preprocessed 
    VAL_RATIO       : ratio between validation and training dataset

3. Run "dataset_gen.py" on the command prompt
   ```sh
    python dataset_gen.py
    ```

### 🎮 Train face-detection

```sh
python main.py
```

### 🧪 Running Tests
```sh
pytest
```

---


## 🗺 Roadmap

> - [X] `ℹ️  Task 1: Implement X`
> - [ ] `ℹ️  Task 2: Refactor Y`
> - [ ] `ℹ️ ...`


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

## 👏 Acknowledgments

> - `ℹ️  List any resources, contributors, inspiration, etc.`

---
