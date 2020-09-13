# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC # Machine Learning Pipeline
# MAGIC 
# MAGIC ** What you will learn:**
# MAGIC * How to Ingest the data into Spark DataFrame.
# MAGIC * How to clean the Data with DataFrame, SQL Query.
# MAGIC * How to create a Machine Learning Pipeline.
# MAGIC * How to train a Machine Learning model.
# MAGIC * How to save & read the model.
# MAGIC * How to make predictions with the model.
# MAGIC 
# MAGIC API Docs : https://spark.apache.org/docs/latest/api.html

# COMMAND ----------

# MAGIC %md
# MAGIC ## The Data
# MAGIC 
# MAGIC The dataset contains bike rental info from 2011 and 2012 in the Capital bikeshare system, plus additional relevant information such as weather.  
# MAGIC 
# MAGIC This dataset is from Fanaee-T and Gama (2013) and is hosted by the <a href="http://archive.ics.uci.edu/ml/datasets/Bike+Sharing+Dataset" target="_blank">UCI Machine Learning Repository</a>.

# COMMAND ----------

# MAGIC %md
# MAGIC ## The Goal
# MAGIC We want to learn to predict bike rental counts (per hour) from information such as day of the week, weather, month, etc.  
# MAGIC 
# MAGIC Having good predictions of customer demand allows a business or service to prepare and increase supply as needed.  

# COMMAND ----------

# MAGIC %md
# MAGIC # Data Ingestion
# MAGIC ## Loading the data
# MAGIC 
# MAGIC We begin by loading our data, which is stored in the CSV format</a>.

# COMMAND ----------

# MAGIC %md ### Types of file systems 
# MAGIC * dbfs:/FileStore/tables/
# MAGIC * hdfs://FileStore/tables/
# MAGIC * s3://FileStore/tables/hour.csv

# COMMAND ----------

from pyspark.sql import SparkSession 
## Used for Local Mac Run
# spark = SparkSession.builder \
#     .master('local[*]') \
#     .appName("CapitalBikeRental") \
#     .getOrCreate()

# COMMAND ----------

display(dbutils.fs.ls("dbfs:/FileStore/tables/"))

# COMMAND ----------

fileName = "/FileStore/tables/hour.csv"

initialDF = (spark.read          # Our DataFrameReader
  .option("header", "true")      # Let Spark know we have a header
  .option("inferSchema", "true") # Infering the schema (it is a small dataset)
  .csv(fileName)                 # Location of our data
  .cache()                       # Mark the DataFrame as cached.
)

initialDF.count()                # Materialize the cache



# COMMAND ----------

# MAGIC %md Check out the schema stracture of the DataFrame.

# COMMAND ----------

initialDF.printSchema()

# COMMAND ----------

# MAGIC %md #### Immutability
# MAGIC 
# MAGIC DataFrame in Spark is **immutable**.
# MAGIC 
# MAGIC What does that mean?
# MAGIC It means that every action we do on DataFrame doesn't change the actual DataFrame!
# MAGIC 
# MAGIC Instead, it creates a new DataFrame.

# COMMAND ----------

# Query Rows 
initialDF.filter(initialDF.instant.between(15, 100))

# COMMAND ----------

# Out[23].show()

# COMMAND ----------

select3 = initialDF.select("atemp","registered","dteday")

# COMMAND ----------

select3.show()

# COMMAND ----------

initialDF.show()

# COMMAND ----------

# MAGIC %md #### You can also Load Data from
# MAGIC * Different DataFormats 
# MAGIC * * JSON, CSV, Parquet, ORC, Avro, LibSVM, Image
# MAGIC * Different DataStores
# MAGIC * * mySQL , Hbase, Hive, cassandra, MongoDB, Kafka, ElasticSearch
# MAGIC 
# MAGIC https://spark.apache.org/docs/latest/sql-data-sources.html

# COMMAND ----------

# MAGIC %md
# MAGIC # Data Understanding
# MAGIC 
# MAGIC According to the <a href="http://archive.ics.uci.edu/ml/datasets/Bike+Sharing+Dataset" target="_blank">UCI ML Repository description</a>, we have the following schema:
# MAGIC 
# MAGIC **Feature columns**:
# MAGIC * **dteday**: date
# MAGIC * **season**: season (1:spring, 2:summer, 3:fall, 4:winter)
# MAGIC * **yr**: year (0:2011, 1:2012)
# MAGIC * **mnth**: month (1 to 12)
# MAGIC * **hr**: hour (0 to 23)
# MAGIC * **holiday**: whether the day was a holiday or not
# MAGIC * **weekday**: day of the week
# MAGIC * **workingday**: `1` if the day is neither a weekend nor holiday, otherwise `0`.
# MAGIC * **weathersit**: 
# MAGIC   * 1: Clear, Few clouds, Partly cloudy, Partly cloudy
# MAGIC   * 2: Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist
# MAGIC   * 3: Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds
# MAGIC   * 4: Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog
# MAGIC * **temp**: Normalized temperature in Celsius. The values are derived via `(t-t_min)/(t_max-t_min)`, `t_min=-8`, `t_max=+39` (only in hourly scale)
# MAGIC * **atemp**: Normalized feeling temperature in Celsius. The values are derived via `(t-t_min)/(t_max-t_min)`, `t_min=-16`, `t_max=+50` (only in hourly scale)
# MAGIC * **hum**: Normalized humidity. The values are divided to 100 (max)
# MAGIC * **windspeed**: Normalized wind speed. The values are divided to 67 (max)
# MAGIC 
# MAGIC **Label columns**:
# MAGIC * **casual**: count of casual users
# MAGIC * **registered**: count of registered users
# MAGIC * **cnt**: count of total rental bikes including both casual and registered
# MAGIC 
# MAGIC **Extraneous columns**:
# MAGIC * **instant**: record index
# MAGIC 
# MAGIC For example, the first row is a record of hour 0 on January 1, 2011---and apparently, 16 people rented bikes around midnight!

# COMMAND ----------

# MAGIC %md
# MAGIC ## EDA : Exploratory DataAnalysis
# MAGIC ###Visualize your data
# MAGIC 
# MAGIC Now that we have preprocessed our features, we can quickly visualize our data to get a sense of whether the features are meaningful.
# MAGIC 
# MAGIC We want to compare bike rental counts versus the hour of the day. 
# MAGIC 
# MAGIC To plot the data:
# MAGIC * Run the cell below
# MAGIC * From the list of plot types, select **Line**.
# MAGIC * Click the **Plot Options...** button.
# MAGIC * By dragging and dropping the fields, set the **Keys** to **hr** and the **Values** to **cnt**.
# MAGIC 
# MAGIC Once you've created the graph, go back and select different **Keys**. For example:
# MAGIC * **cnt** vs. **windspeed**
# MAGIC * **cnt** vs. **month**
# MAGIC * **cnt** vs. **workingday**
# MAGIC * **cnt** vs. **hum**
# MAGIC * **cnt** vs. **temp**
# MAGIC * ...etc.

# COMMAND ----------

display(initialDF)

# COMMAND ----------

# MAGIC %md ### Questions:
# MAGIC   
# MAGIC *   1) At what time Rentals are Low? 
# MAGIC *   2) At what time Rentals are High?
# MAGIC *   3) Which Seasons has more Rental ?
# MAGIC *   4) Which day are rentals High ? ( Working day vs non Working Day)
# MAGIC *   5) What are the Categorical features and Numerical features

# COMMAND ----------

# MAGIC %md
# MAGIC A couple of notes:
# MAGIC * Rentals are low during the night, and they peak in the morning (8 am) and in the early evening (5 pm).  
# MAGIC * Rentals are high during the summer and low in winter.
# MAGIC * Rentals are high on working days vs. non-working days

# COMMAND ----------

# MAGIC %md ### Summary Stats 
# MAGIC * Mean, 
# MAGIC * StandardDeviance, 
# MAGIC * Min ,
# MAGIC * Max , 
# MAGIC * Count

# COMMAND ----------

# Summary statistics (Mean, StandardDeviance, Min ,Max , Count) of Numerical columns
# initialDF.describe().show(5)
initialDF.select("atemp","temp","windspeed").describe().show(5,True)

# COMMAND ----------

# MAGIC %md NOTE : 
# MAGIC * Summary statistics (Mean, StandardDeviance, Min ,Max , Count) of Categorical columns
# MAGIC * output for mean, stddev will be null and
# MAGIC * * min & max values are calculated based on ASCII value of categories

# COMMAND ----------

# MAGIC %md ### SQL

# COMMAND ----------

# MAGIC %md * **weathersit**: 
# MAGIC   * 1: Clear, Few clouds, Partly cloudy, Partly cloudy
# MAGIC   * 2: Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist
# MAGIC   * 3: Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds
# MAGIC   * 4: Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog
# MAGIC 
# MAGIC * **season**: season (1:spring, 2:summer, 3:fall, 4:winter)

# COMMAND ----------

from pyspark.sql.functions import udf
from pyspark.sql.types import StringType
def weathersit_values(r):
    if r == 1 : return "Clear"
    elif r == 2 : return "Mist"
    elif r == 3 : return "Light Rain-Snow"
    else: return "Heavy Rain"

def season_values(r):
    if r == 1 : return "spring"
    elif r == 2 : return "summer"
    elif r == 3 : return "fall"
    else: return "winter"
    
    
weathersitTransorm = udf(weathersit_values, StringType())
seasonTransorm = udf(season_values, StringType())

newDF = initialDF.withColumn("weathersit",weathersitTransorm(initialDF.weathersit)).withColumn("season",seasonTransorm(initialDF.season))

newDF.show()

# COMMAND ----------

newDF.filter(newDF.season != "spring").show()
newDF.filter(newDF.season != "spring").count()

# COMMAND ----------

# MAGIC %md ### SQL Queries 
# MAGIC 
# MAGIC 1. Register the DataFrame as a Table 
# MAGIC 2. Use spark session.sql function 
# MAGIC 3. Returns a new DataFrame

# COMMAND ----------

newDF.createOrReplaceTempView('Bike_Prediction_Table')

# COMMAND ----------

sqlContext.sql("select * from Bike_Prediction_Table where season='summer' and windspeed > 0.4").show(5)

# COMMAND ----------

# maximum booking in each season group in the newDF .
sqlContext.sql('select season, max(cnt) from bike_prediction_table group by season').show()

# COMMAND ----------

display(spark.sql("SELECT season, MAX(temp) as temperature, MAX(hum) as humidity, MAX(windspeed) as windspeed FROM bike_prediction_table GROUP BY season ORDER BY SEASON"))

# COMMAND ----------

# MAGIC %md
# MAGIC # Data Preprocessing
# MAGIC 
# MAGIC So what do we need to do to get our data ready for Machine Learning?
# MAGIC 
# MAGIC **Recall our goal**: We want to learn to predict the count of bike rentals (the `cnt` column).  We refer to the count as our target "label".
# MAGIC 
# MAGIC **Features**: What can we use as features to predict the `cnt` label?  
# MAGIC 
# MAGIC All the columns except `cnt`, and a few exceptions:
# MAGIC * `casual` & `registered`
# MAGIC   * The `cnt` column we want to predict equals the sum of the `casual` + `registered` columns.  We will remove the `casual` and `registered` columns from the data to make sure we do not use them to predict `cnt`.  
# MAGIC   
# MAGIC * `season` and the date column `dteday`: We could keep them, but they are well-represented by the other date-related columns like `yr`, `mnth`, and `weekday`.
# MAGIC * `holiday` and `weekday`: These features are highly correlated with the `workingday` column.
# MAGIC * row index column `instant`: This is a useless column to us.

# COMMAND ----------

# MAGIC %md #####  Warning: Make sure you do not "cheat" by using information you will not have when making predictions*

# COMMAND ----------

# MAGIC %md
# MAGIC Let's drop the columns `instant`, `dteday`, `season`, `casual`, `holiday`, `weekday`, and `registered` from our DataFrame and then review our schema:

# COMMAND ----------

preprocessedDF = initialDF.drop("instant", "dteday", "season", "casual", "registered", "holiday", "weekday")

preprocessedDF.printSchema()

# COMMAND ----------

preprocessedDF.show()

# COMMAND ----------

# MAGIC %md ## Data Cleaning
# MAGIC 
# MAGIC 1. Naïve approach : If the TrainDF contains any null values, we completely drop those rows

# COMMAND ----------

# MAGIC %md ### Mising Values Check

# COMMAND ----------

# checkNullDF = preprocessedDF.na.drop()
# checkNullDF.count()

# COMMAND ----------

# null and empty strings
preprocessedDF.replace('', None).show()
preprocessedDF.replace('', 'null').na.drop(subset='cnt').show()


# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature Assesment
# MAGIC 
# MAGIC As it turns out our features can be divided into two types:
# MAGIC  * **Numeric columns:**
# MAGIC    * `mnth`
# MAGIC    * `temp`
# MAGIC    * `hr`
# MAGIC    * `hum`
# MAGIC    * `atemp`
# MAGIC    * `windspeed`
# MAGIC 
# MAGIC * **Categorical Columns:**
# MAGIC   * `yr`
# MAGIC   * `workingday`
# MAGIC   * `weathersit`
# MAGIC   
# MAGIC We could treat both `mnth` and `hr` as categorical but we would lose the temporal relationships (e.g. 2:00 AM comes before 3:00 AM).

# COMMAND ----------

# MAGIC %md #  Data Processing <a href="https://spark.apache.org/docs/latest/ml-features.html" target="_blank"> (Feature Engineering) </a>
# MAGIC GOAL : To extract the most important features that contribute to the classification.
# MAGIC 
# MAGIC Spark ML API needs our data to be converted in a Spark DataFrame format, 
# MAGIC   - DF → Label (Double) and Features (Vector).
# MAGIC 
# MAGIC 
# MAGIC * Label    → cnt: (Double)
# MAGIC * Features → {"mnth",...,"weathersit"}

# COMMAND ----------

# MAGIC %md
# MAGIC ## StringIndexer
# MAGIC 
# MAGIC For each of the categorical columns, we are going to create one `StringIndexer` where we
# MAGIC   * Set `inputCol` to something like `weathersit`
# MAGIC   * Set `outputCol` to something like `weathersitIndex`
# MAGIC 
# MAGIC This will have the effect of treating a value like `weathersit` not as number 1 through 4, but rather four categories: **light**, **mist**, **medium** & **heavy**, for example.
# MAGIC 
# MAGIC For more information see:
# MAGIC * Scala: <a href="https://spark.apache.org/docs/latest/api/scala/index.html#org.apache.spark.ml.feature.StringIndexer" target="_blank">StringIndexer</a>
# MAGIC * Python: <a href="https://spark.apache.org/docs/latest/api/python/pyspark.ml.html?highlight=stringindexer#pyspark.ml.feature.StringIndexer" target="_blank">StringIndexer</a>

# COMMAND ----------

# MAGIC %md
# MAGIC Before we get started, let's review our current schema:

# COMMAND ----------

preprocessedDF.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC Let's create the first `StringIndexer` for the `workingday` column.
# MAGIC 
# MAGIC After we create it, we can run a sample through the indexer to see how it would affect our `DataFrame`.

# COMMAND ----------

from pyspark.ml.feature import StringIndexer

workingdayStringIndexer = StringIndexer(
  inputCol="workingday", 
  outputCol="workingdayIndex")

# Just for demonstration purposes, we will use the StringIndexer to fit and
# then transform our training data set just to see how it affects the schema
workingdayStringIndexer.fit(preprocessedDF).transform(preprocessedDF).printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC Next we will create the `StringIndexer` for the `yr` column and preview its effect.

# COMMAND ----------

yrStringIndexer = StringIndexer(
  inputCol="yr", 
  outputCol="yrIndex")

yrStringIndexer.fit(preprocessedDF).transform(preprocessedDF).printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC And then create our last `StringIndexer` for the `weathersit` column.

# COMMAND ----------

weathersitStringIndexer = StringIndexer(
  inputCol="weathersit", 
  outputCol="weathersitIndex")

weathersitStringIndexer.fit(preprocessedDF).transform(preprocessedDF).printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC ## VectorAssembler
# MAGIC 
# MAGIC The next step is to assemble the feature columns into a single feature vector.
# MAGIC 
# MAGIC To do that we will use the `VectorAssembler` where we
# MAGIC   * Set `inputCols` to the new list of feature columns
# MAGIC   * Set `outputCol` to `features`
# MAGIC   
# MAGIC   
# MAGIC For more information see:
# MAGIC * Scala: <a href="https://spark.apache.org/docs/latest/api/scala/#org.apache.spark.ml.feature.VectorAssembler" target="_blank">VectorAssembler</a>
# MAGIC * Python: <a href="https://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.feature.VectorAssembler" target="_blank">VectorAssembler</a>

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler

assemblerInputs  = [
  "mnth", "temp", "hr", "hum", "atemp", "windspeed", # Our numerical features
  "yrIndex", "workingdayIndex", "weathersit"]        # Our new categorical features

vectorAssembler = VectorAssembler(
  inputCols=assemblerInputs, 
  outputCol="features")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train/Test Split
# MAGIC 
# MAGIC Our final data preparation step will be to split our dataset into separate training and test sets.
# MAGIC 
# MAGIC Using the `randomSplit()` function, we split the data such that 70% of the data is reserved for training and the remaining 30% for testing. 
# MAGIC 
# MAGIC For more information see:
# MAGIC * Scala: <a href="https://spark.apache.org/docs/latest/api/scala/#org.apache.spark.sql.Dataset" target="_blank">Dataset.randomSplit()</a>
# MAGIC * Python: <a href="https://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.DataFrame.randomSplit" target="_blank">DataFrame.randomSplit()</a>

# COMMAND ----------

trainDF, testDF = preprocessedDF.randomSplit(
  [0.7, 0.3],  # 70-30 split
  seed=42)     # For reproducibility

print("We have %d training examples and %d test examples." % (trainDF.count(), testDF.count()))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Decision Tree
# MAGIC 
# MAGIC Decision trees and their ensembles are popular methods for the machine learning tasks of classification and regression.
# MAGIC 
# MAGIC This is also the last step in our pipeline.
# MAGIC 
# MAGIC We will use the `DecisionTreeRegressor` where we
# MAGIC   * Set `labelCol` to the column that contains our label.
# MAGIC   * Set `seed` to ensure reproducibility.
# MAGIC   * Set `maxDepth` to `10` to control the depth/complexity of the tree.
# MAGIC 
# MAGIC For more information see:
# MAGIC * Scala: <a href="https://spark.apache.org/docs/latest/api/scala/index.html#org.apache.spark.ml.regression.DecisionTreeRegressor" target="_blank">DecisionTreeRegressor</a>
# MAGIC * Python: <a href="https://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.regression.DecisionTreeRegressor" target="_blank">DecisionTreeRegressor</a>

# COMMAND ----------

from pyspark.ml.regression import DecisionTreeRegressor

dtr = (DecisionTreeRegressor()
  .setLabelCol("cnt") # The column of our label
  .setFeaturesCol("features")
  .setSeed(27)        # Some seed value for consistency
  .setMaxDepth(10)    # A guess at the depth of each tree
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create a Machine Learning Pipeline
# MAGIC 
# MAGIC Now let's wrap all of these stages into a Pipeline.

# COMMAND ----------

from pyspark.ml import Pipeline

pipelineTree = Pipeline().setStages([
  workingdayStringIndexer, # categorize workingday
  weathersitStringIndexer, # categorize weathersit
  yrStringIndexer,         # categorize yr
  vectorAssembler,         # assemble the feature vector for all columns
  dtr])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train the model
# MAGIC 
# MAGIC Train the pipeline model to run all the steps in the pipeline.

# COMMAND ----------

pipelineModelTree = pipelineTree.fit(preprocessedDF)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Making Predictions
# MAGIC 
# MAGIC Next, apply the trained pipeline model to the test set.

# COMMAND ----------

predictionsTreeDF = pipelineModelTree.transform(testDF)
predictionsTreeDF.printSchema()

# COMMAND ----------

# Reorder the columns for easier interpretation
reorderedDF = predictionsTreeDF.select("cnt", "prediction", "yr", "yrIndex", "mnth", "hr", "workingday", "workingdayIndex", "weathersit", "weathersitIndex", "temp", "atemp", "hum", "windspeed")

display(reorderedDF)

# COMMAND ----------

# MAGIC %md ## Display Tree and ML Model Parameters

# COMMAND ----------

# MAGIC %md 
# MAGIC Now that we have all of the feature transformations and estimators set up, let's put all of the stages together in the pipline.

# COMMAND ----------

pipelineTree.getStages()

# COMMAND ----------

# If you want to look at what parameter each stage in the pipeline takes.
pipelineTree.getStages()[0].extractParamMap()

# COMMAND ----------

decisionTree = pipelineModelTree.stages[4]
# print(decisionTree.toDebugString)
display(pipelineModelTree.stages[-1])

# COMMAND ----------

# MAGIC %md
# MAGIC ##Evaluate
# MAGIC 
# MAGIC Next, we'll use `RegressionEvaluator` to assess the results. The default regression metric is RMSE.
# MAGIC 
# MAGIC For more information see:
# MAGIC * Scala: <a href="https://spark.apache.org/docs/latest/api/scala/#org.apache.spark.ml.evaluation.RegressionEvaluator" target="_blank">RegressionEvaluator</a>
# MAGIC * Python: <a href="https://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.evaluation.RegressionEvaluator" target="_blank">RegressionEvaluator</a>

# COMMAND ----------

from pyspark.ml.evaluation import RegressionEvaluator

evaluator = RegressionEvaluator().setLabelCol("cnt")

rmse = evaluator.evaluate(predictionsTreeDF)

print("Test RMSE = %f" % rmse)

# COMMAND ----------

# MAGIC %md #### Our RMSE is really high. 
# MAGIC In the next lab, we will cover ways to decrease the RMSE of our model, 
# MAGIC including: 
# MAGIC * cross validation, 
# MAGIC * hyperparameter tuning and  
# MAGIC * ensembles of trees.

# COMMAND ----------

# MAGIC %md
# MAGIC #### Improving our model
# MAGIC 
# MAGIC You are not done yet!  There are several ways we could further improve our model:
# MAGIC * **Expert knowledge** 
# MAGIC * **Better tuning** 
# MAGIC * **Feature engineering**
# MAGIC 
# MAGIC As an exercise: Replace the Decision Tree Algorithm with Random Forest and also with a Gradient Boosted tree, and vary the number of trees and depth of the trees. What do you find?
