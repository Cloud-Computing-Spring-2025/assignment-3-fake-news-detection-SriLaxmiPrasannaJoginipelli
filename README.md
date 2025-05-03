# Fake News Detection using Spark MLlib

This project builds a simple machine learning pipeline to classify news articles as **FAKE** or **REAL** using Apache Spark and its MLlib library.

---

## 📁 Dataset

- **Generated File:** `fake_news_sample.csv`
- **Size:** 500 articles (250 FAKE, 250 REAL)
- **Fields:** `id`, `title`, `text`, `label`

To generate the dataset:

```bash
python Dataset_Generator.py
```
🧪 Task Overview

## Task 1: Load & Basic Exploration
- Load fake_news_sample.csv using Spark and infer schema
- Create a temporary view for SQL-like queries
- Show first 5 rows
- Count total number of articles
- Show distinct labels (FAKE, REAL)
## Output: task1_output.csv

## Task 2: Text Preprocessing
- Convert text to lowercase
- Tokenize using Tokenizer
- Remove stopwords using StopWordsRemover
## Output: task2_output.csv

## Task 3: Feature Extraction
- Extract features using HashingTF and IDF
- Index labels using StringIndexer (FAKE → 0, REAL → 1)
- Assemble into a single feature vector
## Output: task3_output.csv

## Task 4: Model Training
- Split data into 80% training and 20% test sets
- Train a LogisticRegression model
- Predict on the test set
## Output: task4_output.csv

## Task 5: Model Evaluation
- Use MulticlassClassificationEvaluator
- Metrics: Accuracy and F1 Score
## Output: task5_output.csv

## How to Run

1. Install Dependencies

```bash
pip install pyspark faker pandas
```
2. Generate Dataset
```bash
python Dataset_Generator.py
```
3. Run Main Spark Pipeline
```bash
python Main.py
```
- This script will execute all five tasks and save the following output files:

1. task1_output.csv
2. task2_output.csv
3. task3_output.csv
4. task4_output.csv
5. task5_output.csv
