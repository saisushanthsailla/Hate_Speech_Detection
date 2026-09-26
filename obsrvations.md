# Hate Speech Detection using Traditional Machine Learning

## End-to-End Project Process & Technical Reference

---

# 1. Project Objective

The goal of this project was to build a Machine Learning model that classifies social-media text into three categories:

| Class                      | Meaning                                   |
| -------------------------- | ----------------------------------------- |
| **0 — Hate Speech**        | Hateful or discriminatory language        |
| **1 — Offensive Language** | Offensive, abusive, or insulting language |
| **2 — Neither**            | Non-offensive / neutral language          |

The project was intentionally developed using **traditional Machine Learning techniques**, rather than CNNs, Transformers, BERT, or LLMs.

The main objective was to understand how far we could improve a text classification problem using:

* Text preprocessing
* TF-IDF
* Word n-grams
* Character n-grams
* Class balancing
* Logistic Regression
* SVM
* Hyperparameter tuning
* Pipelines
* Error analysis

---

# 2. Overall Project Workflow

The complete workflow was:

```text
Dataset
   ↓
Understand Dataset
   ↓
EDA
   ↓
Identify Target & Features
   ↓
Text Cleaning
   ↓
Train/Test Split
   ↓
Check Class Distribution
   ↓
Baseline TF-IDF
   ↓
Baseline Logistic Regression
   ↓
Evaluate using Macro F1
   ↓
Try Class Balancing
   ↓
Random Oversampling
   ↓
Try Linear SVM
   ↓
Try Balanced SVM
   ↓
Word + Character TF-IDF
   ↓
Create Custom Transformer
   ↓
Pipeline
   ↓
Hyperparameter Tuning
   ↓
Cross Validation
   ↓
Final Model
   ↓
Confusion Matrix
   ↓
Error Analysis
   ↓
Save Model
   ↓
Streamlit Application
   ↓
GitHub
```

---

# 3. Understanding the Dataset

The first step was understanding what the dataset contained.

The dataset consisted of social-media text and a target label.

The three target classes were:

```text
0 → Hate Speech
1 → Offensive Language
2 → Neither
```

Before building a model, it was important to understand:

* Number of samples
* Number of features
* Missing values
* Duplicate records
* Target distribution
* Text characteristics

---

# 4. Why Did We Perform EDA?

EDA helps us understand the problem before applying Machine Learning.

For a text-classification problem, we need to know whether the classes are balanced.

Our training distribution was approximately:

```text
Offensive Language → 77.43%
Neither            → 16.80%
Hate Speech        →  5.77%
```

This immediately revealed an important problem:

## Class imbalance

The model sees far more Offensive Language examples than Hate Speech examples.

If we simply optimize accuracy, a model could perform very well on the majority class while performing poorly on Hate Speech.

Therefore, we decided that:

> Accuracy should not be our only evaluation metric.

---

# 5. Train/Test Split

We separated the dataset into:

```text
Training samples → 19,826
Testing samples  → 4,957
```

The test set was kept separate from model training.

The purpose of the test set is to simulate:

> "What happens when the model receives completely unseen text?"

We used a stratified split so that the class proportions remained approximately consistent between training and testing.

---

# 6. Why Not Train Directly on Raw Text?

Machine Learning algorithms such as Logistic Regression and SVM cannot directly understand text.

For example:

```text
"this is a bad sentence"
```

must be converted into numerical features.

Therefore, we needed a text representation technique.

This is where **TF-IDF** was introduced.

---

# 7. TF-IDF

TF-IDF stands for:

**Term Frequency – Inverse Document Frequency**

It converts text into numerical feature vectors.

The basic idea is:

> Important words in a document should receive higher importance, while extremely common words should receive lower importance.

For example, suppose we have:

```text
tweet 1 → "this is offensive"
tweet 2 → "this is bad"
tweet 3 → "this is harmful"
```

Words such as `"this"` may appear frequently and therefore provide less useful information.

TF-IDF assigns weights based on how useful a term is across the dataset.

---

# 8. Initial TF-IDF Representation

Our initial TF-IDF representation produced:

```text
Training shape → (19826, 18363)
Testing shape  → (4957, 18363)
Vocabulary     → 18363
```

This means:

```text
19,826 documents
        ×
18,363 text features
```

Each tweet became a numerical vector.

---

# 9. Baseline Model

We first built a baseline model.

The purpose of a baseline is:

> Establish a reference performance before trying improvements.

The baseline Logistic Regression produced approximately:

```text
Accuracy        → 0.8943
Macro Precision → 0.8007
Macro Recall    → 0.6420
Macro F1        → 0.6730
```

At first glance, **89.43% accuracy** looks good.

But the Macro F1 was only:

```text
0.6730
```

Why?

Because the model performed poorly on the minority Hate Speech class.

Hate Speech performance was approximately:

```text
Precision → 0.64
Recall    → 0.16
F1        → 0.26
```

This was an important discovery.

---

# 10. Why Macro F1 Became Important

Macro F1 calculates the F1-score separately for each class and then gives every class equal importance.

Conceptually:

```text
Macro F1 =
(F1_Hate + F1_Offensive + F1_Neither) / 3
```

This is useful for our problem because:

```text
Hate Speech       → 5.77%
Offensive         → 77.43%
Neither           → 16.80%
```

If we only looked at accuracy, the majority class could dominate the metric.

Macro F1 forces us to consider:

> How well are we performing across ALL classes?

Therefore, our main optimization metric became:

## Macro F1

---

# 11. First Attempt — Balanced Logistic Regression

We then tried class balancing using Logistic Regression.

The idea was to give greater importance to minority classes.

The result was approximately:

```text
Accuracy        → 0.8578
Macro Precision → 0.6881
Macro Recall    → 0.8035
Macro F1        → 0.7278
```

Notice what happened:

```text
Baseline Macro F1 → 0.6730
Balanced LR       → 0.7278
```

Macro F1 improved significantly.

More importantly, Hate Speech recall improved from approximately:

```text
0.16 → 0.61
```

This demonstrated that class imbalance was a major issue.

---

# 12. Linear SVM

We also experimented with Linear SVM.

The initial Linear SVM produced approximately:

```text
Accuracy    → 0.8939
Macro F1    → 0.7033
```

It performed reasonably well but did not outperform our balanced Logistic Regression approach in Macro F1.

---

# 13. Balanced Linear SVM

We then tried balancing the Linear SVM.

The result was approximately:

```text
Accuracy    → 0.8876
Macro F1    → 0.7277
```

This was useful as an experiment, but it still did not provide a clear improvement over the Logistic Regression approach.

---

# 14. Why Did We Investigate Oversampling?

The dataset was heavily imbalanced.

Instead of only changing the model's class weights, we considered changing the training data itself.

This led us to:

## Random Oversampling

The basic idea:

```text
Original training data

Hate          → 1,145 samples
Offensive     → many more
Neither       → intermediate
```

Random Oversampling duplicates minority-class training samples so that the model gets more exposure to them.

Important:

> Oversampling is performed ONLY on the training data.

The test data must remain untouched.

---

# 15. Why Put Oversampling Inside the Pipeline?

We initially encountered an error when trying to put `RandomOverSampler` into a normal Scikit-learn `Pipeline`.

The reason is that:

```text
RandomOverSampler
```

is not a normal transformer implementing only:

```text
fit()
transform()
```

It performs:

```text
fit_resample()
```

Therefore, we used an **imbalanced-learn Pipeline**.

Conceptually:

```text
TF-IDF
   ↓
Random Oversampling
   ↓
Logistic Regression
```

This is important during cross-validation because each training fold is oversampled independently.

The validation fold remains untouched.

This prevents information leakage.

---

# 16. Word TF-IDF vs Character TF-IDF

We then investigated whether using only word-level features was sufficient.

Word TF-IDF captures patterns such as:

```text
"hate speech"
"stupid person"
"dirty language"
```

But social-media text contains:

* Misspellings
* Slang
* Abbreviations
* Creative spellings
* Variations of offensive words

Therefore, we introduced:

## Character-level TF-IDF

Character n-grams can capture patterns inside words.

For example, different spellings of a word may still share character patterns.

This makes character features particularly useful for noisy social-media text.

---

# 17. Word + Character TF-IDF

Instead of choosing one or the other, we combined them.

Final representation:

```text
Word TF-IDF
      +
Character TF-IDF
```

This allows the model to learn:

### Word-level information

Semantic-ish word combinations and phrases.

### Character-level information

Subword patterns, spelling variations, and noisy text patterns.

---

# 18. Custom WordChar Transformer

To combine the two feature types cleanly, we created a custom transformer.

Conceptually:

```text
Input Text
    │
    ├──────────────→ Word TF-IDF
    │
    └──────────────→ Character TF-IDF
                          │
                          ↓
                    Feature Combination
                          │
                          ↓
                   Combined Features
```

This made the feature extraction reusable during:

* Training
* Cross-validation
* Hyperparameter tuning
* Prediction
* Deployment

---

# 19. Final Machine Learning Pipeline

Our final pipeline became conceptually:

```text
Raw Text
   ↓
Word + Character TF-IDF
   ↓
Random Oversampling
   ↓
Logistic Regression
   ↓
Prediction
```

The advantage of a pipeline is that the complete workflow is treated as one model.

When a new query arrives:

```text
New Tweet
   ↓
Same TF-IDF transformation
   ↓
Trained Logistic Regression
   ↓
Prediction
```

The new query is NOT oversampled.

Oversampling is only a training operation.

---

# 20. Hyperparameter Tuning

After establishing the pipeline, we tuned the important parameters.

We used:

## GridSearchCV

The purpose was to systematically test combinations of parameters.

We tuned:

### Word n-gram range

```text
(1, 2)
```

This means:

```text
Unigrams + Bigrams
```

Example:

```text
"very bad"

very
bad
very bad
```

### Character n-gram range

```text
(3, 6)
```

This allows character sequences of lengths 3 through 6.

### Logistic Regression C

```text
C = 1
```

`C` controls the regularization strength.

---

# 21. Best Parameters

Our tuning produced:

```text
Word n-gram range → (1, 2)
Character n-gram → (3, 6)
C → 1
```

Best cross-validation Macro F1:

```text
0.7438019931485349
```

---

# 22. Final Test Performance

After training the tuned pipeline and evaluating it on the unseen test set:

```text
Accuracy        → 0.89
Macro Precision → 0.73
Macro Recall    → 0.77
Macro F1        → 0.7446
```

Class-wise performance:

| Class     | Precision | Recall |   F1 |
| --------- | --------: | -----: | ---: |
| Hate      |      0.42 |   0.47 | 0.44 |
| Offensive |      0.95 |   0.91 | 0.93 |
| Neither   |      0.81 |   0.92 | 0.86 |

---

# 23. Confusion Matrix

Our final confusion matrix was:

```text
[[174,  84,  28],
 [339, 3291, 208],
 [ 17,  29, 787]]
```

Rows represent:

```text
Actual
```

Columns represent:

```text
Predicted
```

Therefore:

```text
                Predicted
              Hate Offensive Neither

Actual Hate      174     84      28
Actual Offensive 339   3291     208
Actual Neither    17     29     787
```

---

# 24. How to Interpret the Confusion Matrix

### Hate → Hate

```text
174
```

The model correctly identified 174 Hate Speech samples.

### Hate → Offensive

```text
84
```

84 actual Hate Speech samples were classified as Offensive Language.

This was one of our major error types.

### Offensive → Hate

```text
339
```

339 actual Offensive Language samples were classified as Hate Speech.

This is the opposite confusion.

---

# 25. Why Hate vs Offensive Is Difficult

This was one of the most important findings from our error analysis.

Some examples contain offensive words but are not necessarily classified as Hate Speech.

For example, a sentence may contain a slur or insult but the classification depends on:

* Context
* Target of the statement
* Intent
* Usage
* Surrounding words

TF-IDF mainly learns statistical patterns.

It does not truly understand the deeper meaning of a sentence.

Therefore:

```text
Hate Speech ↔ Offensive Language
```

became our major classification challenge.

---

# 26. Error Analysis

Instead of only looking at metrics, we extracted misclassified examples.

We examined:

```text
Actual Hate → Predicted Offensive
```

and:

```text
Actual Offensive → Predicted Hate
```

This helped us understand what the model was getting wrong.

This is important because:

> Model evaluation is not just about a score. We should understand why the model makes mistakes.

---

# 27. Why We Did Not Use Stopword Removal

We discussed whether removing stopwords would improve the model.

We decided not to blindly remove them.

In normal NLP tasks, words such as:

```text
the
is
a
to
```

may provide little information.

But in social-media classification, short contextual words can sometimes contribute to meaning.

Therefore, removing stopwords is not automatically beneficial.

The correct approach would be:

```text
Experiment A → Keep stopwords
Experiment B → Remove stopwords
```

and compare Macro F1.

We did not assume that preprocessing automatically improves performance.

---

# 28. Why We Did Not Remove All Punctuation

We also discussed punctuation.

Punctuation such as:

```text
!
?
```

can sometimes provide useful information in social-media text.

For example:

```text
WHAT?!
```

contains a different pattern from:

```text
what
```

Therefore, aggressive punctuation removal could potentially remove useful signals.

Again, it should be tested rather than assumed to improve performance.

---

# 29. Why Logistic Regression?

Logistic Regression is a strong baseline for high-dimensional sparse text features.

Our TF-IDF representation contains thousands of features.

For example:

```text
19,826 documents
×
many thousands of sparse TF-IDF features
```

Logistic Regression works well in this type of feature space.

Advantages:

* Fast training
* Efficient with sparse matrices
* Interpretable compared with many complex models
* Strong baseline for text classification
* Works well with TF-IDF
* Easy to integrate into a pipeline

---

# 30. Why Not Random Forest / XGBoost as the Final Model?

We experimented with more complex algorithms.

However, complexity does not automatically mean better performance for text classification.

TF-IDF creates a high-dimensional sparse feature matrix.

Linear models such as:

```text
Logistic Regression
Linear SVM
```

are often naturally suited to this type of representation.

Our experiments with more complex approaches did not produce a better Macro F1 than the final Word + Character TF-IDF + Oversampling + Logistic Regression approach.

Therefore, the final model was selected based on the experimental results rather than algorithmic complexity.

---

# 31. Model Serialization

Once the final model was selected, we saved the complete pipeline using Joblib.

The important idea was:

> Save the entire preprocessing + feature extraction + model pipeline together.

This means we don't need to manually reproduce:

```text
TF-IDF transformation
+
oversampling logic
+
Logistic Regression
```

during deployment.

The application can load the trained model and call:

```text
predict()
```

---

# 32. Streamlit Deployment

We then created a Streamlit application.

The user enters text:

```text
User Input
    ↓
Saved Pipeline
    ↓
Word + Character TF-IDF
    ↓
Logistic Regression
    ↓
Prediction
```

The application displays the predicted category.

The three possible outputs are:

```text
Hate Speech
Offensive Language
Neither
```

---

# 33. Why We Used a Pipeline for Deployment

Without a pipeline, we would have to manually remember:

```text
1. Load word vectorizer
2. Transform text
3. Load character vectorizer
4. Transform text
5. Combine features
6. Load model
7. Predict
```

With a pipeline:

```text
model.predict(new_text)
```

The pipeline automatically applies the learned transformations.

This reduces the possibility of preprocessing inconsistencies between training and deployment.

---

# 34. Dependency Management

Before deployment, we checked the versions used in the development environment.

Our environment included:

```text
Python           3.13.15
NumPy            2.1.3
Pandas           2.2.3
SciPy             1.16.3
Scikit-learn      1.6.1
imbalanced-learn  0.14.2
Joblib            1.6.0
```

Matching versions between training and deployment environments helps reduce compatibility problems when loading serialized Scikit-learn models.

---

# 35. Git and GitHub

After completing the project, we prepared it for GitHub.

The Git workflow was:

```text
Project Files
    ↓
git init
    ↓
git add .
    ↓
git commit
    ↓
git branch -M main
    ↓
git remote add origin
    ↓
git push
    ↓
GitHub
```

The repository contains the project code, documentation, dependency information, and deployment files.

---

# 36. Why We Used `.gitignore`

We created a `.gitignore` file to prevent unnecessary or sensitive files from being committed.

Examples:

```text
__pycache__/
.venv/
.env
*.log
.ipynb_checkpoints/
```

We also excluded large model files when appropriate:

```text
*.pkl
```

This prevents accidentally pushing large serialized models to GitHub.

---

# 37. What We Learned from the Project

The biggest learning was that:

> A more complex algorithm does not automatically produce a better Machine Learning model.

We experimented systematically.

The progression was approximately:

```text
Baseline TF-IDF + LR
        ↓
Balanced LR
        ↓
Linear SVM
        ↓
Balanced SVM
        ↓
Oversampling
        ↓
Word + Character TF-IDF
        ↓
Custom Transformer
        ↓
Pipeline
        ↓
Hyperparameter Tuning
        ↓
Final Model
```

Each experiment answered a specific question.

---

# 38. Final Architecture

The final system can be summarized as:

```text
                    USER TEXT
                        │
                        ↓
              ┌─────────────────┐
              │ Feature          │
              │ Transformer      │
              └────────┬────────┘
                       │
              ┌────────┴────────┐
              ↓                 ↓
       Word TF-IDF       Character TF-IDF
              │                 │
              └────────┬────────┘
                       ↓
                Combined Features
                       │
                       ↓
              Random Oversampling
            (Training Only)
                       │
                       ↓
             Logistic Regression
                       │
                       ↓
                  Prediction
                       │
          ┌────────────┼────────────┐
          ↓            ↓            ↓
        Hate       Offensive     Neither
```

---

# 39. Important Interview Questions

You should be able to answer these questions after completing the project.

### Q1. Why did you use Macro F1?

Because the dataset was highly imbalanced and Macro F1 gives equal importance to each class.

---

### Q2. Why was accuracy not enough?

Because Offensive Language represented approximately 77% of the dataset. A model can achieve high accuracy while performing poorly on the minority Hate Speech class.

---

### Q3. Why did you use Random Oversampling?

To give the model more training examples from minority classes and improve their recognition.

---

### Q4. Did you oversample the test set?

No.

Oversampling was applied only to training data.

---

### Q5. Why did you put oversampling inside the pipeline?

To ensure that o
