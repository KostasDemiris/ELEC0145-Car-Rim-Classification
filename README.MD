# Car Classification with YOLO-Cls

This coursework implements a **YOLOv8** classification model to identify 10 different car rim types by fine-tuning using a manually created dataset of 150 images, over 10 different rim types. It includes automated preprocessing to clean filenames, resize images to 224x224, and split data for robust evaluation.

### 1. Prerequisites
```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# or
.venv\Scripts\activate     # Windows

```

### 2. Installation
Install the required dependencies using:

```bash
pip install -r requirements.txt
```

### 3. Project Structure
The source data should be in the following shape:

```plaintext
Coursework 2/
└── Car Dataset Altered/
    ├── Honda Civic Hatchback 2020/
    ├── [The 9 other car rim classes]/
```

### 4. Running the program
Firstly, run the SortImages script to make sure all the formatting is correct, and the data has been correctly seperated into the testing/val/test split.

```python
python SortImages.py
```

Then, run the classifier model training script. Training results, including the confusion matrix, loss curves, and accuracy charts, are saved to:
[runs/classify/Car_Classifier_Final/]

```python
python YoloClassifier.py
```