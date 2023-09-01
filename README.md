# 1. Installation

Before diving into the world of Artificial Intelligence and deep learning, ensure you have all the necessary packages and dependencies installed. 

**Command**:

```bash
python -m pip install -r requirements.txt
```

*Note*: Make sure you're in the correct directory where `requirements.txt` is located. This file will have all the necessary libraries listed for your specific AI project.

# 2. Dataset

Need to download and structure your dataset correctly.

**Download the Dataset**:
- Use [the Google Drive link](https://drive.google.com/file/d/1-USAchKFd5b_URQlKPPsAv9k6VX_nmNI/view?usp=sharing) to download the dataset.

**Folder Structure**:

Organize the dataset in the following manner:

```bash
data_train/
|___ fake/
|    |___ a photo of a flame/
|    |___ a photo of a smoke/
|___ real/
     |___ hard_negative/
     |___ negative/
     |___ positive/
data_test_sub/
train.xlsx
test_sub.xlsx
train.py
test.py
utils.py
requirements.txt
```

# 3. Training

Train your model.

```bash
python train.py
```

# 4. Evaluation
### 4.1. Create an Excel File

Generate an Excel file with the results of your trained model. This file will contain the predictions that your model has computed based on the test data.

```bash
python test.py
```

### 4.2. Submit the Excel File (URL: "http://211.171.175.186:1429/")

Submit 'results.xlsx' to the evaluation server. This server will evaluate the performance.
