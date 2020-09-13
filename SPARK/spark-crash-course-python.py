# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC ## Spark Crash Course
# MAGIC 
# MAGIC 
# MAGIC ** What you will learn today:**
# MAGIC * How to Load/Extract the Data into Spark?
# MAGIC * How to Understand the Data using different Plots?
# MAGIC * Spark Structured and Unstructured APIs.
# MAGIC * Spark DataFrames : Transformations and Actions.
# MAGIC * Spark SQL intro.
# MAGIC 
# MAGIC Ref : https://spark.apache.org/docs/2.3.0/sql-programming-guide.html 
# MAGIC 
# MAGIC [DataFrame FAQs](https://docs.databricks.com/spark/latest/dataframes-datasets/introduction-to-dataframes-python.html#dataframe-faqs)

# COMMAND ----------

# MAGIC %md
# MAGIC ## The Data
# MAGIC 
# MAGIC The dataset contains bike rental info from 2011 and 2012 in the Capital bikeshare system, plus additional relevant information such as weather.  
# MAGIC 
# MAGIC This dataset is from Fanaee-T and Gama (2013) and is hosted by the <a href="http://archive.ics.uci.edu/ml/datasets/Bike+Sharing+Dataset" target="_blank">UCI Machine Learning Repository</a>.

# COMMAND ----------

display(dbutils.fs.ls("dbfs:/FileStore/tables/"))

# COMMAND ----------

# MAGIC %md ### Spark APIs 
# MAGIC 
# MAGIC 1. Create a Spark Context : Its required for executing Operations (Transform + Actions)
# MAGIC 2. Create Spark Session : Driver Process to control Spark APIs
# MAGIC Link - https://spark.apache.org/docs/2.3.0/sql-programming-guide.html#starting-point-sparksession
# MAGIC 
# MAGIC 1. Low Level Unstructured APIs
# MAGIC     * RDDS
# MAGIC 2. Higher Level Structured APIs
# MAGIC     * DataFrame : Table of Data with Rows and Columns
# MAGIC     * SQL 
# MAGIC     * DataSets : TypeSafe 
# MAGIC     
# MAGIC Note - No performance difference between writing DataFrame Transformations or SQL queries, 
# MAGIC They both “compile” to the same underlying plan that we specify in DataFrame code

# COMMAND ----------

from pyspark.sql import SparkSession

# spark = SparkSession \
#     .builder \
#     .appName("Python Spark SQL basic example") \
#     .config("spark.some.config.option", "some-value") \
#     .master("Resourse_manager") \ # (local/yarn)
#     .getOrCreate()

# COMMAND ----------

# MAGIC %md ### DataFrame

# COMMAND ----------

# MAGIC %md 
# MAGIC Why DataFrame?
# MAGIC 
# MAGIC In Apache Spark , DataFrame is a distributed collection of Rows under Named Columns. (like Tables in SQL and Excel Sheet)
# MAGIC 1. Immutable : we can Create DataFrames/RDDs but cannot change it.
# MAGIC      * DF1 --> Transformer --> DF2
# MAGIC      * Lineage can be tracked easily 
# MAGIC 2. Lazy Evalution : Task will not be executed until an action is performed.
# MAGIC 3. Distributed : distribued across the Cluster (partitions)
# MAGIC 4. Resilient : replication of Partitions 
# MAGIC 
# MAGIC NOTE : 
# MAGIC 
# MAGIC * For implicit conversions like converting RDDs to DataFrames 
# MAGIC ```import spark.implicits._```
# MAGIC 
# MAGIC * You can find the similar functionality/ transformations as in Pandas DataFrame.
# MAGIC 
# MAGIC BUT major differences are: 
# MAGIC 1. Running tasks parallel on different nodes
# MAGIC 2. Lazy evaluation 
# MAGIC 3. DFs are immutable in Nature 
# MAGIC 4. Pandas API support more (statistical) operations than PySpark DataFrame. 

# COMMAND ----------

# MAGIC %md Spark ML uses DataFrame from Spark SQL as an ML dataset.
# MAGIC 
# MAGIC Dataframe of data types : 
# MAGIC ```DataFrame [col_TEXT, 
# MAGIC            col_NUMERICAL, 
# MAGIC            col_ARRAY, 
# MAGIC            col_MAP, 
# MAGIC            col_FEATURE_VECTOR, 
# MAGIC            col_TRUE_LABELS, 
# MAGIC            col_PREDICTIONS
# MAGIC           ]```

# COMMAND ----------

# MAGIC %md #### You can also Load Data from
# MAGIC * Different DataFormats 
# MAGIC * * JSON, CSV, Parquet, ORC, Avro, LibSVM, Image
# MAGIC * Different DataStores
# MAGIC * * mySQL , Hbase, Hive, cassandra, MongoDB, Kafka, ElasticSearch
# MAGIC 
# MAGIC https://spark.apache.org/docs/latest/sql-data-sources.html
# MAGIC 
# MAGIC Example - 
# MAGIC * hdfs://FileStore/tables/
# MAGIC * s3://FileStore/tables/hour.csv
# MAGIC * dbfs://FileStore/tables/hour.csv

# COMMAND ----------

# MAGIC %md #### Create Dataframe from a file 
# MAGIC  
# MAGIC ```
# MAGIC // from json file
# MAGIC spark.read.json(path_file.json)
# MAGIC // from text file 
# MAGIC spark.read.textFile("path_file.txt")
# MAGIC // from CSV file 
# MAGIC spark.read.csv("path_file.csv")    
# MAGIC     ```

# COMMAND ----------

fileName = "dbfs:/FileStore/tables/hour.csv"

initialDF = (spark.read          # Our DataFrameReader
  .option("header", "true")      # Let Spark know we have a header
  .option("inferSchema", "true") # Infering the schema (it is a small dataset)
  .csv(fileName)                 # Location of our data
  .cache()                       # Mark the DataFrame as cached.
)

initialDF.count()                # Materialize the cache

initialDF.printSchema()

# COMMAND ----------

initialDF.explain()

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

newDF = initialDF.\
withColumn("weathersit",weathersitTransorm(initialDF.weathersit)).\
withColumn("season",seasonTransorm(initialDF.season))

# COMMAND ----------

# MAGIC %md ### Display Data in different Styles

# COMMAND ----------

initialDF.printSchema()

# COMMAND ----------

# Display first Row
initialDF.first()

# COMMAND ----------

# Display First 5 Rows
initialDF.head(5)

# COMMAND ----------

# Results in Table format
initialDF.show(2,truncate= True)

# COMMAND ----------

# MAGIC %md #### Range

# COMMAND ----------

myRangeDF = spark.range(1000)
myRangeDF.show()
myRangeDF.show(4,truncate= True)

# COMMAND ----------

# MAGIC %md ### Different Ways to Access Columns

# COMMAND ----------

from pyspark.sql.functions import *
from pyspark.sql.types import StringType

#import org.apache.spark.sql.functions.{expr, col, column}

newDF.select(
  expr("season"),
  col("season"),
  column("season"))\
  .show(3)


# COMMAND ----------

# MAGIC %md ### How many Rows and Columns

# COMMAND ----------

# How many rows ?
newDF.count()

# COMMAND ----------

#  How many columns ?
len(newDF.columns)

# COMMAND ----------

# MAGIC %md ### Select Columns

# COMMAND ----------

# How to Select column(s)  
# first 5 rows 
newDF.select("atemp","temp","windspeed").show(5)

# COMMAND ----------

# MAGIC %md ### Distinct numbers

# COMMAND ----------

# Distinct numbers
newDF.select("season").distinct().show()

# Count of Distinct Numbers
newDF.select("windspeed").distinct().count()

# COMMAND ----------

# check df empty
newDF.rdd.isEmpty()

# COMMAND ----------

# MAGIC %md ### Filter Rows

# COMMAND ----------

springFilerDF = newDF.filter(newDF.season != "spring")
springFilerDF.show()

# COMMAND ----------

newDF.filter(newDF.season=="summer") 
newDF.filter(newDF.season.contains("mmer")).show()
newDF.filter(newDF.season.like("%mmer")).show()

# COMMAND ----------

# MAGIC %md ### Type CASTing 
# MAGIC 
# MAGIC Syntax - dataFrame["columnName"].cast(DataType())

# COMMAND ----------

newDF.select("cnt").schema

# COMMAND ----------

from pyspark.sql.types import DoubleType

totalCnt = col("cnt")
newDF.select(totalCnt.cast(DoubleType()))

# COMMAND ----------

# MAGIC %md ### Group By

# COMMAND ----------

newDF.groupby('season').count().show()

# COMMAND ----------

count_seasons = newDF.groupby('season').count()
count_seasons.explain()

# COMMAND ----------

# MAGIC %md ### Order By

# COMMAND ----------

newDF.orderBy(newDF.cnt.desc()).show(5)
newDF.orderBy(newDF.casual.desc(),newDF.cnt ).printSchema()

# COMMAND ----------

# MAGIC %md ### Add New Column or Replace 

# COMMAND ----------

# The withColumn operation will take 2 parameters.
###  Column name which we want add /replace.
###  Expression on column.

# COMMAND ----------

# MAGIC %md 
# MAGIC * **hum**: Normalized humidity. The values are divided to 100 (max)
# MAGIC * **windspeed**: Normalized wind speed. The values are divided to 67 (max)

# COMMAND ----------

newDF.withColumn("hum_new",newDF.hum * 100).show()

# COMMAND ----------

newDF.withColumn("windspeed_notavg", newDF.windspeed * 67)

# COMMAND ----------

# Spark SQL functions lit() and typedLit() are used to add a new column by assigning a literal or constant value to Spark DataFrame. Both functions return Column as return type.

# Both of these are available in Spark by importing org.apache.spark.sql.functions

from pyspark.sql.functions import lit

## New Column - Replace a Column with Constants
newDF.withColumn("cnt", lit(10)).show()

## Add Integer to a column
newDF.select('*', (newDF.cnt + 10000).alias('newCnt')).show()

# COMMAND ----------

newDF.select(col("cnt").alias("totalCnt")).show()
newDF.withColumnRenamed("cnt", "totalCnt").show()

# COMMAND ----------

# MAGIC %md ### Statistical functions
# MAGIC 
# MAGIC Summary Stats 
# MAGIC * Mean, 
# MAGIC * StandardDeviance, 
# MAGIC * Min ,
# MAGIC * Max , 
# MAGIC * Count
# MAGIC 
# MAGIC For more [Statistical-and-mathematical-functions](https://databricks.com/blog/2015/06/02/statistical-and-mathematical-functions-with-dataframes-in-spark.html)
# MAGIC 
# MAGIC 
# MAGIC NOTE : 
# MAGIC 1. Summary statistics (Mean, Standard Deviance, Min ,Max , Count) of Categorical columns
# MAGIC 2. DataFrame with Categorical Columns (StringType) on Describe : 
# MAGIC     -  output : mean, stddev are NULL
# MAGIC     -   min & max values are calculated based on ASCII value of categories

# COMMAND ----------

# Summary statistics (Mean, StandardDeviance, Min ,Max , Count) of Numerical columns
# initialDF.describe().show(5)
initialDF.select("atemp","temp","windspeed").describe().show()

# COMMAND ----------

# MAGIC %md ### Pair wise frequency of categorical columns?
# MAGIC 
# MAGIC 1. First column of each row will be the Distinct values of Season and Column names will be distinct values of Weather Situation
# MAGIC 2. Note : Pair with no occurrences will have zero count in contingency table.

# COMMAND ----------

# Calculate pair wise frequency of categorical columns?
newDF.crosstab('weathersit','season').show()

# COMMAND ----------

newDF.stat.freqItems(["season"],0.4).show(truncate= False)
# df.stat.approxQuantile(...)
# df.stat.bloomFilter(...)
# df.stat.countMinSketch()

# COMMAND ----------

# MAGIC %md ### SQL Queries 
# MAGIC 
# MAGIC 1. Register the DataFrame as a Table 
# MAGIC 2. Use spark session.sql function 
# MAGIC 3. Returns a new DataFrame

# COMMAND ----------

newDF.createOrReplaceTempView('Bike_Prediction_Table')

# COMMAND ----------

sqlContext.sql("select season, cnt from Bike_Prediction_Table ").show()

# COMMAND ----------

sqlContext.sql("select * from Bike_Prediction_Table where season='summer' ").show(5)

# COMMAND ----------

sqlContext.sql("select * from Bike_Prediction_Table where season='summer' and windspeed > 0.4").show()

# COMMAND ----------

# maximum booking in each season group in the newDF .
max_season_df = sqlContext.sql('select season, max(cnt) from bike_prediction_table group by season')

# COMMAND ----------

max_season_df.select("max(cnt)").show()

# COMMAND ----------

# MAGIC %md # DataPreProcessing

# COMMAND ----------

# MAGIC %md
# MAGIC ## Preprocessing the data
# MAGIC 
# MAGIC So what do we need to do to get our data ready for Machine Learning?
# MAGIC 
# MAGIC **Recall our goal**: We want to learn to predict the count of bike rentals (the `cnt` column).  We refer to the count as our target "label".
# MAGIC 
# MAGIC **Features**: What can we use as features to predict the `cnt` label?  
# MAGIC 
# MAGIC All the columns except `cnt`, and a few exceptions:
# MAGIC * `casual` & `registered`
# MAGIC   * The `cnt` column we want to predict equals the sum of the `casual` + `registered` columns.  We will remove the `casual` and `registered` columns from the data to make sure we do not use them to predict `cnt`.  (**Warning: This is a danger in careless Machine Learning.  Make sure you do not "cheat" by using information you will not have when making predictions**)
# MAGIC * `season` and the date column `dteday`: We could keep them, but they are well-represented by the other date-related columns like `yr`, `mnth`, and `weekday`.
# MAGIC * `holiday` and `weekday`: These features are highly correlated with the `workingday` column.
# MAGIC * row index column `instant`: This is a useless column to us.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Cleaning

# COMMAND ----------

# MAGIC %md ### Handling Missing Data
# MAGIC 
# MAGIC 1. Filter Out 
# MAGIC 2. Fill In

# COMMAND ----------

# MAGIC %sh 
# MAGIC cat > movies.csv <<EOF 
# MAGIC name,rating,studio,date 
# MAGIC Avengers Endgame, 5, marvel,1260759144
# MAGIC Batman Vs Superman,4 ,DC,835355664
# MAGIC The Joker ,, DC,835355681
# MAGIC Frozen,4,Disney,835355604
# MAGIC Hitman,,,
# MAGIC EOF 

# COMMAND ----------

# MAGIC %sh ls -alh  /databricks/driver/movies.csv

# COMMAND ----------

path_to_file = "file:///databricks/driver/movies.csv"

moviesDF = (spark.read          # Our DataFrameReader
  .option("header", "true")      # Let Spark know we have a header
  .option("inferSchema", "true") # Infering the schema (it is a small dataset)
  .csv(path_to_file)                 # Location of our data
  .cache()                       # Mark the DataFrame as cached.
)


# COMMAND ----------

moviesDF.printSchema()

# COMMAND ----------

moviesDF.show()

# COMMAND ----------

# MAGIC %md #### 1. Filter OUT
# MAGIC 
# MAGIC [DROP](https://spark.apache.org/docs/2.1.0/api/python/pyspark.sql.html#pyspark.sql.DataFrame.dropna)

# COMMAND ----------

# DROP Any Row with Any NULL Valu
moviesDF.na.drop().show()

# COMMAND ----------

moviesDF.na.drop("all").show()
moviesDF.na.drop("any").show()

# COMMAND ----------

## APPLY to Specific Column
moviesDF.na.drop("all", subset = ['studio','rating']).show()

# COMMAND ----------

# MAGIC %md #### 2. Fill In

# COMMAND ----------

## TYPE Inference of INT in DataFrames

moviesDF.na.fill("1").show()
moviesDF.na.fill(835355604).show()
moviesDF.na.fill("OTHER STUDIO").show()

# COMMAND ----------

## NOTE : FOR PRODUCTION always use the names
moviesDF.select("rating").describe().show()
moviesDF.select(avg(moviesDF['rating'] )).show()
moviesDF.fillna(4.3, subset=['rating']).show()

# COMMAND ----------

# MAGIC %md ### Remove Trivial Data

# COMMAND ----------

# MAGIC %md
# MAGIC Let's drop the columns `instant`, `dteday`, `season`, `casual`, `holiday`, `weekday`, and `registered` from our DataFrame and then review our schema:

# COMMAND ----------

preprocessedDF = initialDF.drop("instant", "dteday", "season", "casual", "registered", "holiday", "weekday")

preprocessedDF.printSchema()

# COMMAND ----------

# MAGIC %md ### dropDuplicates Columns

# COMMAND ----------

newDF.select('Season','windspeed','atemp').count()

# COMMAND ----------

newDF.select('Season','windspeed','atemp').dropDuplicates().count()


# COMMAND ----------

newDF.select('Season','windspeed','atemp').dropDuplicates().show()

# COMMAND ----------

newDF.select("Season").distinct().show()
