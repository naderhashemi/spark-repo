# Databricks notebook source
# Databricks notebook source

#==========================================================================================#
#title           :fpspy.py
#description     :This script/module will installed in python standard libraries location \
#        classes and methods to pull parametes from DB2 and also save
#       	  results to a hive table
#author          :lakshmb
#dateCreated     :2020-05-10
#dateUpdated     :2022-02-07 - environment awarness connections
#version         :POC
#                 Modified by  Srini Madabathula to make it work with environment awarness objects.
#usage           :pyspark fps2py.py OR import fps2py
#notes           :passwords encrypted using base64 library
#===========================================================================================#
#

# COMMAND ----------

import sys
import warnings
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline,PipelineModel
from pyspark.dbutils import DBUtils
import logging
from pyspark.conf import SparkConf
import json
import mlflow 
from pyspark.sql import *
mlflow.set_tracking_uri("Databricks")


HDFS_LOC='/modelerdata/modelresults/'

# COMMAND ----------

# create logger
logger = logging.getLogger('FPSPY')
logger.setLevel(logging.INFO)


# COMMAND ----------

# def getJson():
#   y=dbutils.notebook.entry_point.getDbutils().notebook().getContext().toJson()
#   return str(y)

# id = json.loads(getJson)
# id =json.loads(y)['tags']['user']
#print(id['tags']['user'])
#print(id['tags'])

# dbutils.fs.rm("/Users/H0EF",True)
dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()

# COMMAND ----------

def getSetTrack():
  b = getNotebookName()
  a = dbutils.fs.mkdirs(b)
  return a


def getNotebookName():
  nb = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
  return nb

# COMMAND ----------

getSetTrack()
mlflow.set_experiment(getNotebookName())

# COMMAND ----------

# Create a Spark Session
def createSparkSession():
  spark = SparkSession.builder.appName('getParams').enableHiveSupport().getOrCreate()
  
  spark.sparkContext.setLogLevel('WARN')
  return spark
       

# COMMAND ----------

 # class to get model params


class getModel(object):
  
  def __init__(self,ModelName,Release,Env):
    
    spark = createSparkSession()
    self.ModelBusnsID = ModelName
    self.Release = Release
    self.Env = str(spark._jvm.com.peraton.insight.util.FPSEnvironment.getForThisHost())
    #self.checkModel()
    getUserid()
    getCluster()
            
  def checkModel(self):
    df = getFPSModel(self)
    
    if len(df.head(1)) == 0:
      warnings.warn("Model donot exist in DB")
      
  


# COMMAND ----------

# class to store db2 cpnfig params

def getAttr():
  spark = createSparkSession()
  return [spark._jvm.com.peraton.insight.util.FPSEnvironment.getForThisHost().getJdbcURL()
          ,spark._jvm.com.peraton.insight.util.FPSEnvironment.getForThisHost().getJdbcProperties().get("user"),
          spark._jvm.com.peraton.insight.util.FPSEnvironment.getForThisHost().getJdbcProperties().get("password"),
          spark._jvm.com.peraton.insight.util.FPSEnvironment.getForThisHost().getJdbcProperties().get("driver") ];       
  

# COMMAND ----------

getAttr()

# COMMAND ----------

def getEnvironment():

    spark = createSparkSession()
    return spark._jvm.com.peraton.insight.util.Util.getCurrentDatabricksEnvByServerName()
  

# COMMAND ----------

getEnvironment()

# COMMAND ----------

def getCluster():

    spark = createSparkSession()
    return spark._jvm.com.peraton.insight.util.Util.getCurrentDatabricksClusterName()


# COMMAND ----------

getCluster()

# COMMAND ----------

def getUserid():
  
  spark = createSparkSession()
  userdf = spark.sql("select current_user() as user")
  for i in userdf.collect():
    return i[0] 
  #return userdf.collect()
  

# COMMAND ----------

getUserid()

#/Users/H0EF/fpspy

# COMMAND ----------

def getSnowflakeParams(env):
  
  spark = createSparkSession()
  return {
                   "sfURL": spark._jvm.com.peraton.insight.util.Util.getSnowflakeSparkConnPropertiesByEnv().get("sfURL"),
                   "sfUser": spark._jvm.com.peraton.insight.util.Util.getSnowflakeSparkConnPropertiesByEnv().get("sfUser"),
                   "sfPassword": spark._jvm.com.peraton.insight.util.Util.getSnowflakeSparkConnPropertiesByEnv().get("sfPassword"),
                   "sfDatabase": spark._jvm.com.peraton.insight.util.Util.getSnowflakeSparkConnPropertiesByEnv().get("sfDatabase"),
                   "sfRole": spark._jvm.com.peraton.insight.util.Util.getSnowflakeSparkConnPropertiesByEnv().get("sfRole"),
                   "sfSchema": "FPS_MODELS",
                   "sfWarehouse": spark._jvm.com.peraton.insight.util.Util.getSnowflakeSparkConnPropertiesByEnv().get("sfWarehouse"),
                   "parallelism": "64"
                };

# COMMAND ----------

import fpspy
model=fpspy.getModel('ps_timeliness', 'ModRX.Y', '')
sfOptions=fpspy.getSnowflakeParams(model.Env)
up_dict = {"sfSchema":'fps_mlasr'}
updic = sfOptions.update(up_dict)


# COMMAND ----------

for k,v in up_dict.items():
  print(k,v)

# COMMAND ----------

# getSnowflakeParams(str(spark._jvm.com.peraton.insight.util.FPSEnvironment.getForThisHost()))
getSnowflakeParams('updic')

# COMMAND ----------

def getSnowflakeSource():

  return "net.snowflake.spark.snowflake";


# COMMAND ----------

attr = getAttr()
attr[0] #URL
attr[1] #user
attr[2] #password
attr[3] #driver

# COMMAND ----------

# method that extracts model params as DataFrame from MSTR_MODEL_CONFIG

def getParamsAsSDF(model):
  
  spark = createSparkSession()
  attr=getAttr()
  url=attr[0]
  usr=attr[1]
  pwd=attr[2]
  driver=attr[3]
  
  # connect to DB2 and get the entire table to DataFrame
  try:
    df = spark.read.format('jdbc') \
    .option('url', url) \
    .option('dbtable', spark._jvm.com.peraton.insight.util.FPSEnvironment.getForThisHost().getConfigTableName()) \
    .option('driver', driver) \
    .option('user', usr) \
    .option('password', pwd) \
    .load()
  except:
    raise Exception("Error: Unable to connect to DB2")
  df=df.filter(df.MODEL_BUSNS_ID == model.ModelBusnsID)
  df=df.select('CONFIG_FLD','CONFIG_FLD_VAL')
  return df

# COMMAND ----------

df = (spark.read.format('jdbc')
    .option('url', attr[0])
    .option('dbtable', spark._jvm.com.peraton.insight.util.FPSEnvironment.getForThisHost().getConfigTableName())
    .option('driver', attr[3])
    .option('user', attr[1])
    .option('password', attr[2])
    .load())

# COMMAND ----------

  df=df.select('CONFIG_FLD','CONFIG_FLD_VAL')

# COMMAND ----------

# df = df.filter(df.MODEL_BUSNS_ID.like("SevereSepsisAn"))
display(df)

# COMMAND ----------

# method that extracts model params as Dict from MSTR_MODEL_CONFIG

def getParamsAsDict(model):
  
  # Get params as dataframe
  df=getParamsAsSDF(model)

  # Convert DF to Dict
  newrdd = df.rdd.map(lambda x : (x[0],x[1]))
  dict = newrdd.collectAsMap()
  return dict


# COMMAND ----------

newrdd = df.rdd.map(lambda x: (x[0],x[1]))
dict = newrdd.collectAsMap()


# COMMAND ----------

# method that extracts model params from FPS_MODEL

def getFPSModel(model):
  
  spark = createSparkSession()
  attr=getAttr()
  url=attr[0]
  usr=attr[1]
  pwd=attr[2]
  driver=attr[3]

  try:
    
    df = spark.read.format('jdbc') \
              .option('url', url) \
              .option('dbtable', spark._jvm.com.peraton.insight.util.FPSEnvironment.getForThisHost().getResultsTableName("FPS_MODEL")) \
              .option('driver', driver) \
              .option('user', usr) \
              .option('password', pwd) \
              .load()
  except:
    logger.debug("Error: Unable to connect to DB2")
    sys.exit(1)

  df=df.filter(df.MODEL_BUSNS_ID == model.ModelBusnsID)
  return df

# COMMAND ----------

df_fps = (spark.read.format('jdbc')
    .option('url', attr[0])
    .option('dbtable', spark._jvm.com.peraton.insight.util.FPSEnvironment.getForThisHost().getResultsTableName("FPS_MODEL"))
    .option('driver', attr[3])
    .option('user', attr[1])
    .option('password', attr[2])
    .load())

# COMMAND ----------

# method to save DF to Hive

def send(model,df,tblsfx):
  
  sfxList=['npisuspect','claimsatrisk','claimsatriskptd','reportcard','reportcardsna','supplementalrc']
  tblsfx=tblsfx.lower()
  if tblsfx not in sfxList:
    raise Exception('Error: Table name suffix {} not in the list'.format(tblsfx))
            
  spark=createSparkSession()
  #SNOWFLAKE_SOURCE_NAME = "net.snowflake.spark.snowflake"
  SNOWFLAKE_SOURCE_NAME = getSnowflakeSource()
  sfOptions = getSnowflakeParams(model.Env)
  tblName="fps_models.{0}_{1}".format(model.ModelBusnsID,tblsfx)
  df.write.format(SNOWFLAKE_SOURCE_NAME).options(**sfOptions).option("dbtable", tblName).mode("overwrite").save()
  logger.info("Completed sending data to table {}".format(tblName))

# COMMAND ----------

# method to persist spark dataframe to specified table in hive


def persistDF(df,tblName, env):

       spark=createSparkSession()
       SNOWFLAKE_SOURCE_NAME = getSnowflakeSource()
       sfOptions = getSnowflakeParams(env)
       df.write.format(SNOWFLAKE_SOURCE_NAME).options(**sfOptions).option("dbtable", tblName.strip().lower()).mode("overwrite").save()
       logger.info("Completed persisting data to table {}".format(tblName))

# COMMAND ----------

# method to persist spark dataframe to specified table in hive


def persistDF2(df, dbSchema, tblName, env):

       spark=createSparkSession()
       SNOWFLAKE_SOURCE_NAME = getSnowflakeSource()
       sfOptions = getSnowflakeParams(env)
       df.write.format(SNOWFLAKE_SOURCE_NAME).options(**sfOptions).option("sfSchema", dbSchema.strip().lower()).option("dbtable", tblName.strip().lower()).mode("overwrite").save()
       logger.info("Completed persisting data to table {}".format(tblName))



def save(model,pipeline):

       pipeType=pipeline.uid.split('_')[0]
       if pipeType == 'PipelineModel':
           pipedirname='trainedpipeline'
       elif pipeType == 'Pipeline':
           pipedirname='pipeline'

       modelname=model.ModelBusnsID
       out_hdfs_path='{0}/{1}/{2}'.format(HDFS_LOC,modelname,pipedirname)
       pipeline.write().overwrite().save(out_hdfs_path)


# COMMAND ----------

# load a saved pipeline/pipelinemodel


def load(model,pipelineType):
  
  modelname=model.ModelBusnsID
  pipeType=pipelineType.lower()
  
  if pipeType == 'pipelinemodel':
    pipeLoc='{0}/{1}/trainedpipeline'.format(HDFS_LOC,modelname)
    try:
      loadedPipeline=PipelineModel.load(pipeLoc)
    except:
      logger.debug('ERROR: No PipelineModel exist for loading')
      return
  elif pipeType == 'pipeline':
    pipeLoc='{0}/{1}/pipeline'.format(HDFS_LOC,modelname)
    try:
      loadedPipeline=Pipeline.load(pipeLoc)
    except:
      logger.debug('ERROR: No Pipeline exist for loading')
      return
  else:
    raise Exception('Error: Invalid Pipeline Type')
    return loadedPipeline

# COMMAND ----------

#' query snowflake table to a dataframe
def sqlHive(model,query):
  return queryWH(model,query)

# COMMAND ----------

def queryWH(model,query):
  spark=createSparkSession()
  SNOWFLAKE_SOURCE_NAME = getSnowflakeSource()
  sfOptions = getSnowflakeParams(model.Env)
  df=spark.read.format("snowflake").options(**sfOptions).option("query", query).load()
  logger.info("Completed queryWH data to table {}")
  return df 

# COMMAND ----------

def referenceDate (model,  dt=None):

 
  spark = createSparkSession()
  dbutils = DBUtils(spark)
  

  try:
     refdate = dbutils.widgets.get("enddate")
     #logger.debug('Obtained refdate from DBUTILS')
  except:
     refdate =  None
     
  if refdate is None:
    refdate = dt
       
  return refdate

# COMMAND ----------


