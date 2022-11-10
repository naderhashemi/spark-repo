# Databricks notebook source
## Pyspark script to train and save an ML model to assist
##  in the creation of the provider education 90 day flag.
## Last Updated: 04/06/2022


from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark import *
from pyspark.sql import SparkSession
import time
import sys

from pyspark import SparkContext, SparkConf
from pyspark import HiveContext, SQLContext
from datetime import datetime, timezone, timedelta
import pandas as pd

from pyspark.ml.clustering import KMeans
from pyspark.ml.regression import GBTRegressor,RandomForestRegressor,LinearRegression
from pyspark.ml.evaluation import ClusteringEvaluator,RegressionEvaluator
from pyspark.ml.feature import StringIndexer,OneHotEncoder,VectorAssembler,VectorIndexer,Bucketizer,StandardScaler
from pyspark.ml import Pipeline
from pyspark.ml import PipelineModel
from pyspark.ml.linalg import Vectors
from pyspark.mllib.evaluation import RegressionMetrics
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

# COMMAND ----------

# Creates a Spark session and context

spark = SparkSession.builder.appName('pe90_model').getOrCreate()
spark.sparkContext.setLogLevel('WARN')

# COMMAND ----------

# DBTITLE 1,Connect to Snowflake
user = dbutils.secrets.get("snowflake", "snowflake-user")
password = dbutils.secrets.get("snowflake", "snowflake-pwd")
sf_connection = dict(sfUrl= "cms_fps.us-east-1-gov.privatelink.snowflakecomputing.com:443",
sfUser= user,
sfPassword = password,
sfDatabase = "FPS_DEV",
sfSchema = "FPS_MLASR",
sfWarehouse = "WH_LRG")

SNOWFLAKE_SOURCE_NAME = "net.snowflake.spark.snowflake"

# COMMAND ----------

# DBTITLE 1,Snowflake Queries
parta_qry = '''
SELECT * 
FROM "FPS_MLASR"."PE90_MODEL_DATASET_PARTA"
'''

partb_qry = '''
SELECT *
FROM "FPS_MLASR"."PE90_MODEL_DATASET"
WHERE clm_src_type = 'B'
'''

dme_qry = '''
SELECT *
FROM "FPS_MLASR"."PE90_MODEL_DATASET"
WHERE clm_src_type = 'DME'
'''

pe90_qry = '''
SELECT *
FROM "FPS_MLASR"."PE90_MODEL_DATASET"
'''

# COMMAND ----------

# DBTITLE 1,Load Snowflake tables(previously lived in Hive)
pe90_model_dataset = (sqlContext.read
                  .format(SNOWFLAKE_SOURCE_NAME)
                  .options(**sf_connection)
                  .option("query",pe90_qry)
                  .load())
pe90_parta_model = (sqlContext.read
                  .format(SNOWFLAKE_SOURCE_NAME)
                  .options(**sf_connection)
                  .option("query",parta_qry)
                  .load())
pe90_partb_model = (sqlContext.read
                  .format(SNOWFLAKE_SOURCE_NAME)
                  .options(**sf_connection)
                  .option("query",partb_qry)
                  .load())
pe90_dme_model = (sqlContext.read
                  .format(SNOWFLAKE_SOURCE_NAME)
                  .options(**sf_connection)
                  .option("query",dme_qry)
                  .load())

# COMMAND ----------

pe90_parta_model= pe90_parta_model.na.replace('', 'EMPTY').withColumnRenamed("clm_sbmt_chrg_amt","label")
pe90_dme_model= pe90_dme_model.na.replace('', 'EMPTY').withColumnRenamed("clm_sbmt_chrg_amt","label")
pe90_partb_model= pe90_partb_model.na.replace('', 'EMPTY').withColumnRenamed("clm_sbmt_chrg_amt","label")
pe90_model_dataset= pe90_model_dataset.na.replace('', 'EMPTY').withColumnRenamed("clm_sbmt_chrg_amt","label")

# COMMAND ----------

# DBTITLE 1,Transformation: Part-B and DME
# Function to create all the data transformations for the Part B and DME model.
# 	String Indexers convert a String to an index based on the frequency of the string.
# 	Bucketizer will replace a value with a bucket based on the splits and frequency.
# 	Vector Assembler organizes all the output columns of the Indexers into a vector to pass to the model.
# 	Vector Indexer will take a vector and replace it with an index based on fequency and uniqueness.



def create_tranforms():
  
  state_cd_indexer = StringIndexer(inputCol="STATE_CD", outputCol="state_cd_idx").setHandleInvalid("skip")
  state_cd_encoder = OneHotEncoder(inputCols =["state_cd_idx"], outputCols=["state_cd_enc"])

  clm_line_hcpcs_indexer = StringIndexer(inputCol="CLM_LINE_HCPCS_CD", outputCol="clm_line_hcpcs_cd_idx").setHandleInvalid("skip")
  clm_line_hcpcs_encoder = OneHotEncoder(inputCols =["clm_line_hcpcs_cd_idx"], outputCols=["clm_line_hcpcs_cd_enc"])
  
  hcpcs_1_mdfr_cd_indexer = StringIndexer(inputCol="HCPCS_1_MDFR_CD", outputCol="hcpcs_1_mdfr_cd_idx").setHandleInvalid("skip")
  hcpcs_1_mdfr_cd_encoder = OneHotEncoder(inputCols =["hcpcs_1_mdfr_cd_idx"], outputCols=["hcpcs_1_mdfr_cd_enc"])

  hcpcs_2_mdfr_cd_indexer = StringIndexer(inputCol="HCPCS_2_MDFR_CD", outputCol="hcpcs_2_mdfr_cd_idx").setHandleInvalid("skip")
  hcpcs_2_mdfr_cd_encoder = OneHotEncoder(inputCols =["hcpcs_2_mdfr_cd_idx"], outputCols=["hcpcs_2_mdfr_cd_enc"])

  clm_prncpl_dgns_cd_indexer = StringIndexer(inputCol="CLM_PRNCPL_DGNS_CD", outputCol="clm_prncpl_dgns_cd_idx").setHandleInvalid("skip")
  clm_prncpl_dgns_cd_encoder = OneHotEncoder(inputCols =["clm_prncpl_dgns_cd_idx"], outputCols=["clm_prncpl_dgns_cd_enc"])
  
  clm_dgns_1_cd_indexer = StringIndexer(inputCol="CLM_DGNS_1_CD", outputCol="clm_dgns_1_cd_idx").setHandleInvalid("skip")
  clm_dgns_1_cd_encoder = OneHotEncoder(inputCols =["clm_dgns_1_cd_idx"], outputCols=["clm_dgns_1_cd_enc"])

  clm_dgns_2_cd_indexer = StringIndexer(inputCol="CLM_DGNS_2_CD", outputCol="clm_dgns_2_cd_idx").setHandleInvalid("skip")
  clm_dgns_2_cd_encoder = OneHotEncoder(inputCols =["clm_dgns_2_cd_idx"], outputCols=["clm_dgns_2_cd_enc"])
  
  clm_dgns_3_cd_indexer = StringIndexer(inputCol="CLM_DGNS_3_CD", outputCol="clm_dgns_3_cd_idx").setHandleInvalid("skip")
  clm_dgns_3_cd_encoder = OneHotEncoder(inputCols =["clm_dgns_3_cd_idx"], outputCols=["clm_dgns_3_cd_enc"])

  clm_dgns_4_cd_indexer = StringIndexer(inputCol="CLM_DGNS_4_CD", outputCol="clm_dgns_4_cd_idx").setHandleInvalid("skip")
  clm_dgns_4_cd_encoder = OneHotEncoder(inputCols =["clm_dgns_4_cd_idx"], outputCols=["clm_dgns_4_cd_enc"])

  clm_dgns_5_cd_indexer = StringIndexer(inputCol="CLM_DGNS_5_CD", outputCol="clm_dgns_5_cd_idx").setHandleInvalid("skip")
  clm_dgns_5_cd_encoder = OneHotEncoder(inputCols =["clm_dgns_5_cd_idx"], outputCols=["clm_dgns_5_cd_enc"])

  clm_prcdr_cd_indexer = StringIndexer(inputCol="CLM_PRCDR_CD", outputCol="clm_prcdr_cd_idx").setHandleInvalid("skip")
  clm_prcdr_cd_encoder = OneHotEncoder(inputCols =["clm_prcdr_cd_idx"], outputCols=["clm_prcdr_cd_enc"])

  input_cols = ['state_cd_enc','clm_line_hcpcs_cd_enc','hcpcs_1_mdfr_cd_enc','hcpcs_2_mdfr_cd_enc','clm_prncpl_dgns_cd_enc',
                'clm_dgns_1_cd_enc','clm_dgns_2_cd_enc','clm_dgns_3_cd_enc','clm_dgns_4_cd_enc','clm_dgns_5_cd_enc','clm_prcdr_cd_enc']

  assembler = VectorAssembler().setInputCols(input_cols).setOutputCol("features")

  return [state_cd_indexer,state_cd_encoder,clm_line_hcpcs_indexer,clm_line_hcpcs_encoder,
          hcpcs_1_mdfr_cd_indexer,hcpcs_1_mdfr_cd_encoder,hcpcs_2_mdfr_cd_indexer,
          hcpcs_2_mdfr_cd_encoder,clm_prncpl_dgns_cd_indexer,clm_prncpl_dgns_cd_encoder,
          clm_dgns_1_cd_indexer,clm_dgns_1_cd_encoder,clm_dgns_2_cd_indexer,
          clm_dgns_2_cd_encoder,clm_dgns_3_cd_indexer,clm_dgns_3_cd_encoder,
          clm_dgns_4_cd_indexer,clm_dgns_4_cd_encoder,clm_dgns_5_cd_indexer,
          clm_dgns_5_cd_encoder,clm_prcdr_cd_indexer,clm_prcdr_cd_encoder,assembler]


# COMMAND ----------

# DBTITLE 1,Transformation: Part-A
# Function to create all the data transformations for the Part A model.
# 	String Indexers convert a String to an index based on the frequency of the string.
# 	Vector Assembler organizes all the output columns of the Indexers into a vector to pass to the model.


def create_transforms_pta():
  
  state_cd_indexer = StringIndexer(inputCol="STATE_CD", outputCol="state_cd_idx").setHandleInvalid("skip")
  state_cd_encoder = OneHotEncoder(inputCols =["state_cd_idx"], outputCols=["state_cd_enc"])
    
  clm_line_hcpcs_indexer = StringIndexer(inputCol="CLM_LINE_HCPCS_CD", outputCol="clm_line_hcpcs_cd_idx").setHandleInvalid("skip")
  clm_line_hcpcs_encoder = OneHotEncoder(inputCols =["clm_line_hcpcs_cd_idx"], outputCols=["clm_line_hcpcs_cd_enc"])
    
  hcpcs_1_mdfr_cd_indexer = StringIndexer(inputCol="HCPCS_1_MDFR_CD", outputCol="hcpcs_1_mdfr_cd_idx").setHandleInvalid("skip")
  hcpcs_1_mdfr_cd_encoder = OneHotEncoder(inputCols =["hcpcs_1_mdfr_cd_idx"], outputCols=["hcpcs_1_mdfr_cd_enc"])

  hcpcs_2_mdfr_cd_indexer = StringIndexer(inputCol="HCPCS_2_MDFR_CD", outputCol="hcpcs_2_mdfr_cd_idx").setHandleInvalid("skip")
  hcpcs_2_mdfr_cd_encoder = OneHotEncoder(inputCols =["hcpcs_2_mdfr_cd_idx"], outputCols=["hcpcs_2_mdfr_cd_enc"])
    
  clm_prncpl_dgns_cd_indexer = StringIndexer(inputCol="CLM_PRNCPL_DGNS_CD", outputCol="clm_prncpl_dgns_cd_idx").setHandleInvalid("skip")
  clm_prncpl_dgns_cd_encoder = OneHotEncoder(inputCols =["clm_prncpl_dgns_cd_idx"], outputCols=["clm_prncpl_dgns_cd_enc"])

  clm_dgns_1_cd_indexer = StringIndexer(inputCol="CLM_DGNS_1_CD", outputCol="clm_dgns_1_cd_idx").setHandleInvalid("skip")
  clm_dgns_1_cd_encoder = OneHotEncoder(inputCols =["clm_dgns_1_cd_idx"], outputCols=["clm_dgns_1_cd_enc"])
    
  clm_dgns_2_cd_indexer = StringIndexer(inputCol="CLM_DGNS_2_CD", outputCol="clm_dgns_2_cd_idx").setHandleInvalid("skip")
  clm_dgns_2_cd_encoder = OneHotEncoder(inputCols =["clm_dgns_2_cd_idx"], outputCols=["clm_dgns_2_cd_enc"])
    
  clm_dgns_3_cd_indexer = StringIndexer(inputCol="CLM_DGNS_3_CD", outputCol="clm_dgns_3_cd_idx").setHandleInvalid("skip")
  clm_dgns_3_cd_encoder = OneHotEncoder(inputCols =["clm_dgns_3_cd_idx"], outputCols=["clm_dgns_3_cd_enc"])
    
  clm_dgns_4_cd_indexer = StringIndexer(inputCol="CLM_DGNS_4_CD", outputCol="clm_dgns_4_cd_idx").setHandleInvalid("skip")
  clm_dgns_4_cd_encoder = OneHotEncoder(inputCols =["clm_dgns_4_cd_idx"], outputCols=["clm_dgns_4_cd_enc"])
    
  clm_dgns_5_cd_indexer = StringIndexer(inputCol="CLM_DGNS_5_CD", outputCol="clm_dgns_5_cd_idx").setHandleInvalid("skip")
  clm_dgns_5_cd_encoder = OneHotEncoder(inputCols =["clm_dgns_5_cd_idx"], outputCols=["clm_dgns_5_cd_enc"])
    
  clm_prcdr_1_cd_indexer = StringIndexer(inputCol="CLM_PRCDR_1_CD", outputCol="clm_prcdr_1_cd_idx").setHandleInvalid("skip")
  clm_prcdr_1_cd_encoder = OneHotEncoder(inputCols =["clm_prcdr_1_cd_idx"], outputCols=["clm_prcdr_1_cd_enc"])
    
  clm_prcdr_2_cd_indexer = StringIndexer(inputCol="CLM_PRCDR_2_CD", outputCol="clm_prcdr_2_cd_idx").setHandleInvalid("skip")
  clm_prcdr_2_cd_encoder = OneHotEncoder(inputCols =["clm_prcdr_2_cd_idx"], outputCols=["clm_prcdr_2_cd_enc"])
    
  clm_prcdr_3_cd_indexer = StringIndexer(inputCol="CLM_PRCDR_3_CD", outputCol="clm_prcdr_3_cd_idx").setHandleInvalid("skip")
  clm_prcdr_3_cd_encoder = OneHotEncoder(inputCols =["clm_prcdr_3_cd_idx"], outputCols=["clm_prcdr_3_cd_enc"])
    
  clm_prcdr_4_cd_indexer = StringIndexer(inputCol="CLM_PRCDR_4_CD", outputCol="clm_prcdr_4_cd_idx").setHandleInvalid("skip")
  clm_prcdr_4_cd_encoder = OneHotEncoder(inputCols =["clm_prcdr_4_cd_idx"], outputCols=["clm_prcdr_4_cd_enc"])
    
  clm_prcdr_5_cd_indexer = StringIndexer(inputCol="CLM_PRCDR_5_CD", outputCol="clm_prcdr_5_cd_idx").setHandleInvalid("skip")
  clm_prcdr_5_cd_encoder = OneHotEncoder(inputCols =["clm_prcdr_5_cd_idx"], outputCols=["clm_prcdr_5_cd_enc"])

  input_cols = ['state_cd_enc','clm_line_hcpcs_cd_enc','hcpcs_1_mdfr_cd_enc','hcpcs_2_mdfr_cd_enc',
                'clm_prncpl_dgns_cd_enc','clm_dgns_1_cd_enc','clm_dgns_2_cd_enc','clm_dgns_3_cd_enc',
                'clm_dgns_4_cd_enc','clm_dgns_5_cd_enc','clm_prcdr_1_cd_enc','clm_prcdr_2_cd_enc',
                'clm_prcdr_3_cd_enc','clm_prcdr_4_cd_enc','clm_prcdr_5_cd_enc']

  assembler = VectorAssembler().setInputCols(input_cols).setOutputCol("features")
    
  return [state_cd_indexer,state_cd_encoder,clm_line_hcpcs_indexer,clm_line_hcpcs_encoder,hcpcs_1_mdfr_cd_indexer,
          hcpcs_1_mdfr_cd_encoder,hcpcs_2_mdfr_cd_indexer,hcpcs_2_mdfr_cd_encoder,clm_prncpl_dgns_cd_indexer,
          clm_prncpl_dgns_cd_encoder,clm_dgns_1_cd_indexer,clm_dgns_1_cd_encoder,clm_dgns_2_cd_indexer,clm_dgns_2_cd_encoder,
          clm_dgns_3_cd_indexer,clm_dgns_3_cd_encoder,clm_dgns_4_cd_indexer,clm_dgns_4_cd_encoder,clm_dgns_5_cd_indexer,
          clm_dgns_5_cd_encoder,clm_prcdr_1_cd_indexer,clm_prcdr_1_cd_encoder,clm_prcdr_2_cd_indexer,clm_prcdr_2_cd_encoder,
          clm_prcdr_3_cd_indexer,clm_prcdr_3_cd_encoder,clm_prcdr_4_cd_indexer,clm_prcdr_4_cd_encoder,
          clm_prcdr_5_cd_indexer,clm_prcdr_5_cd_encoder,assembler]




# COMMAND ----------

# DBTITLE 1,Test: Vectors and Pipline
############ --  Nader: Do Not Delete this cell 


# state_cd_indexer = StringIndexer(inputCol = "STATE_CD", outputCol = "state_cd_idx")
# onhotencoder_state = OneHotEncoder(inputCols = ["state_cd_idx"], outputCols = ["state_cd_enc"])
# pipeline=Pipeline(stages=[state_cd_indexer,onhotencoder_state])
# df_transformed = pipeline.fit(pe90_parta_model).transform(pe90_parta_model)

# COMMAND ----------

#modified for dataiku
# (train_data, test_data) = pe90_dme_model.randomSplit([0.7, 0.3],seed=1234)
# evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse")
# lr = LinearRegression(featuresCol = 'features', labelCol='label', maxIter=100, regParam=0.1, elasticNetParam=1.0)
# transforms = create_tranforms()
# stages_input = transforms + [lr]

  
# linReg_pipeline = Pipeline(stages=stages_input)
# linReg_model = linReg_pipeline.fit(train_data)
# test_data = test_data.drop('prediction')
# modelOutcome = linReg_model.transform(test_data)
# result = modelOutcome.select(col('prediction').cast("double"), col('label').cast("double")).rdd.map(lambda x: (x['prediction'],x['label']))


# COMMAND ----------

# reg_eval=RegressionMetrics(result)

# COMMAND ----------

# type(reg_eval)

# COMMAND ----------

# Function to evaluate any trained regressor from pyspark.ml.
#   It will calculate the Root Mean Squared Error, Mean Squared Error
#   R Squared, and the Mean Absolute Error.
# Returns: Dict with all evalualation results

def eval_Regressor(test_data,regressorModel):
  
  
  test_data = test_data.drop('prediction')

  modelOutcome = regressorModel.transform(test_data)

  result = modelOutcome.select(col('prediction').cast("double"), col('label').cast("double"))\
          .rdd.map(lambda x: (x['prediction'],x['label']))

  reg_eval = RegressionMetrics(result)
  #print(regressorModel.getEstimator().explainParams())
  #print("Root Mean Squared Error: " + str(reg_eval.rootMeanSquaredError))
  #print("Mean Squared Error: " + str(reg_eval.meanSquaredError))
  #print("R^2: " + str(reg_eval.r2))
  #print("Mean Absolute Error: " + str(reg_eval.meanAbsoluteError))
  eval_results = {"Root_Mean_Sqaured_Error":reg_eval.rootMeanSquaredError,
                  "Mean_Squared_Error":reg_eval.meanSquaredError,
                  "R_Squared": reg_eval.r2,
                  "Mean_Absolute_Error": reg_eval.meanAbsoluteError}
  print(type(result))
  print(type(reg_eval))
  return eval_results


# COMMAND ----------

def run_linReg_regression(df,model_type):
  
  mtyp = "pe90_" + model_type
  print(mtyp)
  algorithm = "Linear Regression"
  print(algorithm)
  train_ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
  print(train_ts)
  
  (train_data, test_data) = df.randomSplit([0.7, 0.3],seed=1234)
  train_size = train_data.count()
  
  
  if (model_type == 'PTA'):
    transforms = create_transforms_pta()
    #path = 'dbfs:/FileStore/tables/pe90/pta'
    path = 'dbfs:/ml/pe90_model/pta'
    #modelPath = 'dbfs:/FileStore/tables/pe90/partA/trainedpipeline'
    modelPath = 'dbfs:/ml/pe90_model/parta/trainedpipeline'
   
  elif (model_type == 'PTB'):
    transforms = create_tranforms()
    path = 'dbfs:/ml/pe90_model/ptb'
    #path = 'dbfs:/FileStore/tables/pe90/ptb'
    #modelPath = 'dbfs:/FileStore/tables/pe90/partB/trainedpipeline'
    modelPath = 'dbfs:/ml/pe90_model/partb/trainedpipeline'
  

  else:
    transforms = create_tranforms()
    #path = 'dbfs:/FileStore/tables/pe90/dme'
    path = 'dbfs:/ml/pe90_model/dme'
    #modelPath = 'dbfs:/FileStore/tables/pe90/partdme/trainedpipeline'
    modelPath = 'dbfs:/ml/pe90_model/partdme/trainedpipeline'

 
  
  evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse")
  start = time.process_time()
  
  lr = LinearRegression(featuresCol = 'features', labelCol='label', maxIter=100, regParam=0.1, elasticNetParam=1.0)
  stages_input = transforms + [lr]
  
  linReg_pipeline = Pipeline(stages=stages_input)
  linReg_model = linReg_pipeline.fit(train_data)
  
  #linReg_model.save(modelPath)
  
  linReg_model.write().overwrite().save(modelPath)
  
  
  
  end_time = (time.process_time() - start)
  #linReg_train_time = round((end - start)/60,2)
  print(end_time)
    
  lr_evalResults = eval_Regressor(test_data,linReg_model)
    
  data = [(mtyp,algorithm,train_ts,end_time,train_size,lr_evalResults['Root_Mean_Sqaured_Error'],lr_evalResults['Mean_Squared_Error'],lr_evalResults['R_Squared'],lr_evalResults['Mean_Absolute_Error'])]
  columns = ["Model_name","Algorithm","Train_ts","Train_time","Train_size","Root_Mean_Sqaured_Error","Mean_Squared_Error","R_Squared","Mean_Absolute_Error"]
  eval_df = spark.createDataFrame(data,columns)
  eval_df.display()
  eval_df.write.format('delta').mode('overwrite').save(path)
   

# COMMAND ----------

# pe90_dme_model.write.mode("overwrite").format('delta').saveAsTable("fps_mlasr.pe90_dme_db")
pe90_dme_model.createOrReplaceTempView('dmss')
pe90_parta_model.createOrReplaceTempView('ass')
pe90_partb_model.createOrReplaceTempView('bss')

# COMMAND ----------

# MAGIC %sql
# MAGIC select count(*) cnt from ass;

# COMMAND ----------

# MAGIC %sql
# MAGIC create or replace temporary view pe90_ass_vw as (
# MAGIC select * from ass);

# COMMAND ----------

pe90_ass = spark.table('pe90_ass_vw')

# COMMAND ----------

run_linReg_regression(pe90_dme_model,'DME')

# COMMAND ----------

# spark.catalog.clearCache()

# COMMAND ----------

# run_linReg_regression(pe90_parta_model,'PTA')
run_linReg_regression(pe90_ass,'PTA')

# COMMAND ----------


run_linReg_regression(pe90_partb_model,'PTB')

# COMMAND ----------

pe90_partb_model.count()

# COMMAND ----------

dme_df = spark.read.format("delta").load("dbfs:/ml/pe90_model/dme")
pta_df = spark.read.format("delta").load("dbfs:/ml/pe90_model/pta")
ptb_df = spark.read.format("delta").load("dbfs:/ml/pe90_model/ptb")
mrg_dme_pta = dme_df.unionByName(pta_df)
final_mrg_eval = mrg_dme_pta.unionByName(ptb_df)

# COMMAND ----------

final_mrg_eval.write.format('delta').mode('overwrite').save('dbfs:/ml/pe90_model/master_eval')

# COMMAND ----------

# MAGIC %sql
# MAGIC DROP TABLE IF EXISTS fps_mlasr.master_evaluation;

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE TABLE fps_mlasr.master_evaluation
# MAGIC USING DELTA
# MAGIC LOCATION 'dbfs:/ml/pe90_model/master_eval';

# COMMAND ----------

# (final_mrg_eval.write.format("snowflake")
#  .options(**sf_connection)
#  .option("dbtable", "pe90_model_master_eval")
#  .mode('append')
#  .save())

# COMMAND ----------


