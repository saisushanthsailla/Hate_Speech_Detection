# 🛡️ Hate Speech Detection using Machine Learning

A **Natural Language Processing (NLP)** project that classifies social-media text into three categories:

* 🔴 **Hate Speech**
* 🟠 **Offensive Language**
* 🟢 **Neither**

The project focuses on building an interpretable **traditional Machine Learning pipeline** rather than using deep learning, transformers, or large language models.

---

## 🚀 Project Overview

Social-media platforms contain large amounts of user-generated content, making automatic detection of harmful language an important NLP task.

In this project, I developed an end-to-end text classification system that takes raw text as input and predicts whether it belongs to **Hate Speech, Offensive Language, or Neither**.

The project covers the complete Machine Learning workflow:

```text
Raw Text
   ↓
Text Preprocessing
   ↓
Train/Test Split
   ↓
Word + Character TF-IDF
   ↓
Class Imbalance Handling
   ↓
Random Oversampling
   ↓
Logistic Regression
   ↓
Hyperparameter Tuning
   ↓
Error Analysis
   ↓
Model Evaluation
   ↓
Streamlit Application
```

---

## 🎯 Objective

The primary objective is to build a Machine Learning model capable of distinguishing between different types of harmful or non-harmful language while paying particular attention to the minority **Hate Speech** class.

Since the dataset is highly imbalanced, **accuracy alone is not sufficient** for evaluating the model.

Therefore, **Macro F1-score, Macro Precision, Macro Recall, and class-wise performance** were considered during model development.

---

## 📊 Dataset

The dataset contains three target classes:

| Class                      | Description                                       |
| -------------------------- | ------------------------------------------------- |
| **0 — Hate Speech**        | Text containing hateful or discriminatory content |
| **1 — Offensive Language** | Abusive, insulting, or offensive language         |
| **2 — Neither**            | Non-offensive or neutral content                  |

### Class Distribution

The dataset was significantly imbalanced:

| Class              | Percentage |
| ------------------ | ---------: |
| Offensive Language | **77.43%** |
| Neither            | **16.80%** |
| Hate Speech        |  **5.77%** |

Because Hate Speech represents only a small portion of the dataset, special attention was given to minority-class performance.

---

## 🧹 Text Processing

The text data was processed before feature extraction.

The workflow included handling the raw tweet text and preparing it for TF-IDF feature extraction.

Rather than relying only on individual words, the final model uses both:

### Word-level TF-IDF

Captures meaningful word patterns and combinations.

Example:

```text
"hate speech"
"very offensive"
"stupid person"
```

### Character-level TF-IDF

Captures character patterns inside words.

This can help with:

* Misspelled words
* Abbreviations
* Slang
* Variations of offensive words
* Social-media specific language

The final feature representation combines:

```text
Word TF-IDF + Character TF-IDF
```

---

## ⚖️ Handling Class Imbalance

The original dataset contained a large majority of **Offensive Language** samples and comparatively fewer **Hate Speech** samples.

To improve minority-class learning, **Random Oversampling** was experimented with.

Random Oversampling increases the representation of minority-class samples in the training data.

Importantly, resampling was performed **inside the Machine Learning pipeline**, ensuring that the test data remained untouched.

```text
Training Data
      ↓
TF-IDF
      ↓
Random Oversampling
      ↓
Logistic Regression
```

The test set was never oversampled.

---

## 🤖 Model Development

Several Machine Learning approaches were experimented with during the project.

The experiments included:

* Baseline Logistic Regression
* Balanced Logistic Regression
* Linear SVM
* Balanced Linear SVM
* Random Oversampling
* Word + Character TF-IDF
* Tree-based models
* Hyperparameter tuning

After comparing the experiments, the final model selected for deployment was:

### ⭐ Word + Character TF-IDF + Random Oversampling + Logistic Regression

---

## 🔧 Hyperparameter Tuning

Grid Search with cross-validation was used to tune the final pipeline.

The selected parameters were:

```text
Word n-gram range : (1, 2)
Character n-gram  : (3, 6)
Logistic Regression C : 1
```

Best cross-validation Macro F1:

```text
0.7438
```

---

## 📈 Final Model Performance

The final model achieved the following performance on the unseen test set:

| Metric          |      Score |
| --------------- | ---------: |
| Accuracy        |   **0.89** |
| Macro Precision |   **0.73** |
| Macro Recall    |   **0.77** |
| **Macro F1**    | **0.7446** |

### Classification Report

| Class              | Precision | Recall | F1-score |
| ------------------ | --------: | -----: | -------: |
| Hate Speech        |      0.42 |   0.47 |     0.44 |
| Offensive Language |      0.95 |   0.91 |     0.93 |
| Neither            |      0.81 |   0.92 |     0.86 |

The **Macro F1-score** was used as an important evaluation metric because it gives equal importance to all three classes rather than allowing the majority class to dominate the evaluation.

---

## 🔍 Error Analysis

A confusion matrix and misclassified examples were analyzed to understand where the model struggled.

A major challenge was distinguishing:

```text
Hate Speech ↔ Offensive Language
```

Many examples contain offensive words but require contextual understanding to determine whether they should be classified as Hate Speech or Offensive Language.

This demonstrates an important limitation of traditional TF-IDF-based models: they primarily learn statistical text patterns rather than deeper semantic context.

---

## 🧩 Custom Feature Transformer

A custom transformer was developed to combine:

```text
Word-level TF-IDF
+
Character-level TF-IDF
```

This allowed both feature extraction methods to be integrated into the Machine Learning workflow and used consistently during training and inference.

---

## 🔗 Final Pipeline

The final prediction pipeline can be represented as:

```text
Input Text
    │
    ↓
Word + Character Feature Extraction
    │
    ↓
Random Oversampling
    │
    ↓
Logistic Regression
    │
    ↓
Predicted Class
```

For a new user query, the model applies the same feature transformation learned during training before generating the prediction.

---

## 🌐 Streamlit Application

A Streamlit web application was developed to provide an interactive interface for the trained model.

The application allows users to:

* Enter custom text
* Generate predictions
* View the predicted category
* View prediction confidence
* Understand the three classification categories
* Explore model information

### Application Preview

Add your screenshot here:

```text
screenshots/app.png
```

You can display it in the README using:

```markdown
![Hate Speech Detection App](screenshots/app.png)
```

---

## 🛠️ Technologies Used

### Programming Language

* Python

### Machine Learning

* Scikit-learn
* Imbalanced-learn

### NLP

* TF-IDF
* Word N-grams
* Character N-grams
* Text Classification

### Data Processing

* NumPy
* Pandas

### Model Persistence

* Joblib

### Deployment / UI

* Streamlit

### Development Environment

* Google Colab
* Local Python Environment
* Git
* GitHub

---

## 📁 Project Structure

```text
Hate_Speech_Detection/
│
├── app.py
├── word_char_features.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── screenshots/
│   └── app.png
│
└── hate_speech_word_char_lr.pkl
```

> The trained model file may be hosted separately if it exceeds GitHub's file-size limits.

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/saisushanthsailla/Hate_Speech_Detection.git
```

Move into the project directory:

```bash
cd Hate_Speech_Detection
```

Create and activate a virtual environment:

```bash
python -m venv hate_speech
```

Windows:

```bash
hate_speech\Scripts\activate
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Streamlit Application

After installing the dependencies:

```bash
streamlit run app.py
```

The application will open in your browser.

---

## 🧪 Example

### Input

```text
The weather is beautiful today.
```

### Prediction

```text
Neither
```

The application also displays the model's confidence for each class.

---

## 💡 Key Learnings

Through this project, I gained practical experience in:

* NLP text classification
* TF-IDF feature engineering
* Word and character n-grams
* Handling imbalanced datasets
* Random oversampling
* Machine Learning pipelines
* Logistic Regression
* Hyperparameter tuning
* Cross-validation
* Macro F1 evaluation
* Confusion matrix analysis
* Error analysis
* Custom Scikit-learn transformers
* Model serialization
* Streamlit application development
* Git and GitHub

---

## ⚠️ Limitations

This model is based on traditional Machine Learning and TF-IDF features.

Therefore, it has limitations when dealing with:

* Sarcasm
* Context-dependent language
* Implicit hate speech
* New slang
* Highly ambiguous statements
* Complex semantic relationships

For example, simply detecting offensive words does not necessarily mean that a sentence represents Hate Speech.

The error analysis showed that distinguishing **Hate Speech from Offensive Language** remains challenging.

---

## 🔮 Future Improvements

Potential future improvements include:

* Advanced text preprocessing
* Feature engineering
* Ensemble Machine Learning models
* Better class-balancing strategies
* Threshold optimization
* More extensive hyperparameter tuning
* Transformer-based NLP models
* BERT/RoBERTa-based classification
* Multilingual hate-speech detection
* Continuous model monitoring

These approaches could be explored as a next step beyond the traditional Machine Learning scope of this project.

---

## 👨‍💻 Author

**Sai Sushanth Sailla**

