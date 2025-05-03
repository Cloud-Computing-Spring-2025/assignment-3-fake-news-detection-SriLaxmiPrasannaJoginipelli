from pyspark.sql import SparkSession
from pyspark.sql.functions import lower, col
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF, StringIndexer, VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Start Spark session
spark = SparkSession.builder.appName("FakeNewsDetection").getOrCreate()

# ========== Task 1: Load & Basic Exploration ==========
df = spark.read.csv("fake_news_sample.csv", header=True, inferSchema=True)
df.createOrReplaceTempView("news_data")

# Show first 5 rows
df.show(5)

# Count articles and distinct labels
print("Total articles:", df.count())
df.select("label").distinct().show()

# Save sample output
df.limit(5).toPandas().to_csv("output/task1_output.csv", index=False)

# ========== Task 2: Text Preprocessing ==========
# Lowercase and tokenize
df_lower = df.withColumn("text", lower(col("text")))

tokenizer = Tokenizer(inputCol="text", outputCol="words")
words_data = tokenizer.transform(df_lower)

# Remove stopwords
remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
cleaned_data = remover.transform(words_data)

# Save output
cleaned_data.select("id", "title", "filtered_words", "label") \
    .toPandas().to_csv("output/task2_output.csv", index=False)

# ========== Task 3: Feature Extraction ==========
# TF-IDF
hashingTF = HashingTF(inputCol="filtered_words", outputCol="rawFeatures", numFeatures=10000)
featurized_data = hashingTF.transform(cleaned_data)

idf = IDF(inputCol="rawFeatures", outputCol="features")
idf_model = idf.fit(featurized_data)
rescaled_data = idf_model.transform(featurized_data)

# Index labels
indexer = StringIndexer(inputCol="label", outputCol="label_index")
data_indexed = indexer.fit(rescaled_data).transform(rescaled_data)

# Save output
data_indexed.select("id", "filtered_words", "features", "label_index") \
    .toPandas().to_csv("output/task3_output.csv", index=False)

# ========== Task 4: Model Training ==========
# Train/test split
train_data, test_data = data_indexed.randomSplit([0.8, 0.2], seed=42)

# Logistic Regression
lr = LogisticRegression(featuresCol="features", labelCol="label_index")
lr_model = lr.fit(train_data)

# Predict
predictions = lr_model.transform(test_data)

# Save predictions
predictions.select("id", "title", "label_index", "prediction") \
    .toPandas().to_csv("output/task4_output.csv", index=False)

# ========== Task 5: Evaluate the Model ==========
evaluator_acc = MulticlassClassificationEvaluator(labelCol="label_index", predictionCol="prediction", metricName="accuracy")
evaluator_f1 = MulticlassClassificationEvaluator(labelCol="label_index", predictionCol="prediction", metricName="f1")

accuracy = evaluator_acc.evaluate(predictions)
f1_score = evaluator_f1.evaluate(predictions)

# Save evaluation
with open("output/task5_output.csv", "w") as f:
    f.write("Metric,Value\n")
    f.write(f"Accuracy,{accuracy:.2f}\n")
    f.write(f"F1 Score,{f1_score:.2f}\n")

# Done!
spark.stop()
