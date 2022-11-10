# Databricks notebook source
## Pyspark script to load a trained ML model to assist
##  in the creation of the provider education 90 day flag.
## Last Updated: 04/13/2022

import time
import sys
from datetime import *
import numpy as np
import argparse
from pyspark import *
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark import SparkContext, SparkConf
from pyspark import HiveContext, SQLContext
from pyspark.sql import *
#from pyspark.ml.pipeline import PipelineModel
from pyspark.ml import PipelineModel
from pyspark.ml.feature import VectorAssembler
from scipy.stats import ttest_ind_from_stats

# COMMAND ----------



# COMMAND ----------

# Creates a Spark session and context

spark = SparkSession.builder.appName('pe90_model').getOrCreate()
spark.sparkContext.setLogLevel('WARN')

# COMMAND ----------

# DBTITLE 1,Snowflake Connection
user = dbutils.secrets.get("snowflake", "snowflake-user")
password = dbutils.secrets.get("snowflake", "snowflake-pwd")
sf_connection = dict(sfUrl= "cms_fps.us-east-1-gov.privatelink.snowflakecomputing.com:443",
sfUser= user,
sfPassword = password,
sfDatabase = "FPS_MTE",
sfSchema = "FPS_MLASR",                   
sfWarehouse = "WH_LRG")
SNOWFLAKE_SOURCE_NAME = "net.snowflake.spark.snowflake"

# COMMAND ----------

# DBTITLE 1,Snowflake Queries
pe90_ucm_educated_cases = '''
SELECT * 
FROM "FPS_MTE"."FPS_MLASR"."PE90_UCM_EDUCATED_CASES"
'''

tpe_educated_cases = '''
SELECT *
FROM "FPS_MTE"."FPS_DW"."TPE_EDUCATED_CASES"
'''

pe90_ucm_most_current_education = '''
SELECT * 
FROM "FPS_MTE"."FPS_MLASR"."PE90_UCM_MOST_CURRENT_EDUCATION"
'''

pe90_tpe_most_current_education = '''
SELECT *
FROM "FPS_MTE"."FPS_MLASR"."PE90_TPE_MOST_CURRENT_EDUCATION"
'''

pe90_most_current_education = '''
SELECT * 
FROM "FPS_MTE"."FPS_MLASR"."PE90_MOST_CURRENT_EDUCATION"
'''

pe90_alleducated_with_maxalertdate = '''
SELECT *
FROM "FPS_MTE"."FPS_MLASR"."PE90_ALLEDUCATED_WITH_MAXALERTDATE"
'''

pe90_educatedlist = '''
SELECT *
FROM "FPS_MTE"."FPS_MLASR"."PE90_EDUCATEDLIST"
'''

pe90_testing_data_before = '''
SELECT *
FROM "FPS_MTE"."FPS_MLASR"."PE90_TESTING_DATA_BEFORE"
'''
pe90_testing_data_after = '''
SELECT *
FROM "FPS_MTE"."FPS_MLASR"."PE90_TESTING_DATA_AFTER"
'''

pe90_testing_data_before_parta = '''
SELECT *
FROM "FPS_MTE"."FPS_MLASR"."PE90_TESTING_DATA_BEFORE_PARTA"
'''

pe90_testing_data_after_parta = '''
SELECT *
FROM "FPS_MTE"."FPS_MLASR"."PE90_TESTING_DATA_AFTER_PARTA";
'''
pe90_asrlist = ''' 
SELECT * FROM "FPS_MTE"."FPS_MLASR"."PE90_ASRLIST";
'''

fps_asrpt = '''
SELECT * FROM "FPS_MTE"."FPS_DW"."FPS_ASRPT";
'''

fps_asrpt_alert_asctn = '''
SELECT * FROM "FPS_MTE"."FPS_DW"."FPS_ASRPT_ALERT_ASCTN";
'''
fps_alert = '''
SELECT * FROM "FPS_MTE"."FPS_DW"."FPS_ALERT";
'''

fps_model = '''
SELECT * FROM "FPS_MTE"."FPS_DW"."FPS_MODEL";
'''

# COMMAND ----------

# DBTITLE 1,Load Snowflake tables(previously lived in Hive)
pe90_ucm_edu = (sqlContext.read
                  .format(SNOWFLAKE_SOURCE_NAME)
                  .options(**sf_connection)
                  .option("query",pe90_ucm_educated_cases)
                  .load())

tpe_educated_cases = (sqlContext.read
                  .format(SNOWFLAKE_SOURCE_NAME)
                  .options(**sf_connection)
                  .option("query",tpe_educated_cases)
                  .load())

pe90_ucm_most_curr_edu = (sqlContext.read
                  .format(SNOWFLAKE_SOURCE_NAME)
                  .options(**sf_connection)
                  .option("query",pe90_ucm_most_current_education)
                  .load())
pe90_tpe_most_curr_edu = (sqlContext.read
                  .format(SNOWFLAKE_SOURCE_NAME)
                  .options(**sf_connection)
                  .option("query",pe90_tpe_most_current_education)
                  .load())
pe90_most_curr_edu = (sqlContext.read
                  .format(SNOWFLAKE_SOURCE_NAME)
                  .options(**sf_connection)
                  .option("query",pe90_most_current_education)
                  .load())
pe90_alledu_maxale = (sqlContext.read
                  .format(SNOWFLAKE_SOURCE_NAME)
                  .options(**sf_connection)
                  .option("query",pe90_alleducated_with_maxalertdate)
                  .load())
pe90_educatedlist = (sqlContext.read
                  .format(SNOWFLAKE_SOURCE_NAME)
                  .options(**sf_connection)
                  .option("query",pe90_educatedlist)
                  .load())
pe90_testing_data_bef = (sqlContext.read
                  .format(SNOWFLAKE_SOURCE_NAME)
                  .options(**sf_connection)
                  .option("query",pe90_testing_data_before)
                  .load())
pe90_testing_data_aft = (sqlContext.read
                  .format(SNOWFLAKE_SOURCE_NAME)
                  .options(**sf_connection)
                  .option("query",pe90_testing_data_after)
                  .load())
pe90_testing_bef_parta = (sqlContext.read
                  .format(SNOWFLAKE_SOURCE_NAME)
                  .options(**sf_connection)
                  .option("query",pe90_testing_data_before_parta)
                  .load())

pe90_testing_aft_parta = (sqlContext.read
                  .format(SNOWFLAKE_SOURCE_NAME)
                  .options(**sf_connection)
                  .option("query",pe90_testing_data_after_parta)
                  .load())

pe90_asrlist = (sqlContext.read
                  .format(SNOWFLAKE_SOURCE_NAME)
                  .options(**sf_connection)
                  .option("query",pe90_asrlist)
                  .load())
  
                
fps_asrpt =  (sqlContext.read
                  .format(SNOWFLAKE_SOURCE_NAME)
                  .options(**sf_connection)
                  .option("query",fps_asrpt)
                  .load())

fps_asrpt_alert_asctn = (sqlContext.read
                  .format(SNOWFLAKE_SOURCE_NAME)
                  .options(**sf_connection)
                  .option("query",fps_asrpt_alert_asctn)
                  .load())

fps_alert = (sqlContext.read
                  .format(SNOWFLAKE_SOURCE_NAME)
                  .options(**sf_connection)
                  .option("query",fps_alert)
                  .load())

fps_model = (sqlContext.read
                  .format(SNOWFLAKE_SOURCE_NAME)
                  .options(**sf_connection)
                  .option("query",fps_model)
                  .load())

# COMMAND ----------

# DBTITLE 1,Prep Test Data


before_educ_ptb = pe90_testing_data_bef.na.replace('', 'EMPTY').filter(col('clm_src_type')=='B')\
.select('clm_blg_prvdr_npi_num','clm_line_from_dt','STATE_CD','clm_line_hcpcs_cd','hcpcs_1_mdfr_cd'\
          ,'hcpcs_2_mdfr_cd','clm_prncpl_dgns_cd','clm_dgns_1_cd','clm_dgns_2_cd','clm_dgns_3_cd'\
          ,'clm_dgns_4_cd','clm_dgns_5_cd','clm_prcdr_cd','clm_sbmt_chrg_amt')

  
before_educ_dme = pe90_testing_data_bef.na.replace('', 'EMPTY').filter(col('clm_src_type')=='DME')\
.select('clm_blg_prvdr_npi_num','clm_line_from_dt','STATE_CD','clm_line_hcpcs_cd','hcpcs_1_mdfr_cd'\
          ,'hcpcs_2_mdfr_cd','clm_prncpl_dgns_cd','clm_dgns_1_cd','clm_dgns_2_cd','clm_dgns_3_cd'\
          ,'clm_dgns_4_cd','clm_dgns_5_cd','clm_prcdr_cd','clm_sbmt_chrg_amt')

before_educ_pta = pe90_testing_bef_parta.na.replace('', 'EMPTY')\
.select('clm_blg_prvdr_npi_num','current_education_date','STATE_CD','clm_line_from_dt','clm_sbmt_chrg_amt'\
,'clm_line_hcpcs_cd','hcpcs_1_mdfr_cd','hcpcs_2_mdfr_cd','clm_prncpl_dgns_cd','clm_dgns_1_cd'\
,'clm_dgns_2_cd','clm_dgns_3_cd','clm_dgns_4_cd','clm_dgns_5_cd','clm_prcdr_cd','clm_prcdr_1_cd','clm_prcdr_2_cd'\
,'clm_prcdr_3_cd','clm_prcdr_4_cd','clm_prcdr_5_cd')
  
  
  
after_educ_ptb = pe90_testing_data_aft.na.replace('', 'EMPTY').filter(col('clm_src_type')=='B')\
  .select('clm_blg_prvdr_npi_num','clm_line_from_dt','STATE_CD','clm_line_hcpcs_cd','hcpcs_1_mdfr_cd'\
          ,'hcpcs_2_mdfr_cd','clm_prncpl_dgns_cd','clm_dgns_1_cd','clm_dgns_2_cd','clm_dgns_3_cd'\
          ,'clm_dgns_4_cd','clm_dgns_5_cd','clm_prcdr_cd','clm_sbmt_chrg_amt')
  
after_educ_dme = pe90_testing_data_aft.na.replace('', 'EMPTY').filter(col('clm_src_type')=='DME')\
  .select('clm_blg_prvdr_npi_num','clm_line_from_dt','STATE_CD','clm_line_hcpcs_cd','hcpcs_1_mdfr_cd'\
          ,'hcpcs_2_mdfr_cd','clm_prncpl_dgns_cd','clm_dgns_1_cd','clm_dgns_2_cd','clm_dgns_3_cd'\
          ,'clm_dgns_4_cd','clm_dgns_5_cd','clm_prcdr_cd','clm_sbmt_chrg_amt')
  
after_educ_pta = pe90_testing_aft_parta.na.replace('', 'EMPTY')\
  .select('clm_blg_prvdr_npi_num','current_education_date','STATE_CD','clm_line_from_dt','clm_sbmt_chrg_amt'\
          ,'clm_line_hcpcs_cd','hcpcs_1_mdfr_cd','hcpcs_2_mdfr_cd','clm_prncpl_dgns_cd','clm_dgns_1_cd'\
          ,'clm_dgns_2_cd','clm_dgns_3_cd','clm_dgns_4_cd','clm_dgns_5_cd','clm_prcdr_cd','clm_prcdr_1_cd','clm_prcdr_2_cd'\
          ,'clm_prcdr_3_cd','clm_prcdr_4_cd','clm_prcdr_5_cd')
  
  

# COMMAND ----------

before_educ_ptb = before_educ_ptb.toDF(*[c.upper() for c in before_educ_ptb.columns])
before_educ_dme = before_educ_dme.toDF(*[c.upper() for c in before_educ_dme.columns])
before_educ_pta = before_educ_pta.toDF(*[c.upper() for c in before_educ_pta.columns])
after_educ_ptb = after_educ_ptb.toDF(*[c.upper() for c in after_educ_ptb.columns])
after_educ_dme = after_educ_dme.toDF(*[c.upper() for c in after_educ_dme.columns])
after_educ_pta = after_educ_pta.toDF(*[c.upper() for c in after_educ_pta.columns])

# COMMAND ----------

regression_model_ptb = PipelineModel.load('dbfs:/ml/pe90_model/partb/trainedpipeline')
regression_model_dme = PipelineModel.load('dbfs:/ml/pe90_model/partdme/trainedpipeline')
regression_model_pta = PipelineModel.load('dbfs:/ml/pe90_model/parta/trainedpipeline')

# COMMAND ----------

# Estimating Pristine Provider Submitted Charge Amount for Each Claim for Before - Part B
before_educ_estimate_ptb = regression_model_ptb.transform(before_educ_ptb)
before_educ_estimate_ptb.createOrReplaceTempView("before_educ_estimate_ptb")
# Estimating Pristine Provider Submitted Charge Amount for Each Claim for Before - DME
before_educ_estimate_dme = regression_model_dme.transform(before_educ_dme)
before_educ_estimate_dme.createOrReplaceTempView("before_educ_estimate_dme")
# Estimating Pristine Provider Submitted Charge Amount for Each Claim for Before - Part A
before_educ_estimate_pta = regression_model_pta.transform(before_educ_pta)
before_educ_estimate_pta.createOrReplaceTempView("before_educ_estimate_pta")
# Estimating Pristine Provider Submitted Charge Amount for Each Claim for After - Part B
after_educ_estimate_ptb = regression_model_ptb.transform(after_educ_ptb)
after_educ_estimate_ptb.createOrReplaceTempView("after_educ_estimate_ptb")
# Estimating Pristine Provider Submitted Charge Amount for Each Claim for After - DME
after_educ_estimate_dme = regression_model_dme.transform(after_educ_dme)
after_educ_estimate_dme.createOrReplaceTempView("after_educ_estimate_dme")
# Estimating Pristine Provider Submitted Charge Amount for Each Claim for After - Part A
after_educ_estimate_pta = regression_model_pta.transform(after_educ_pta)
after_educ_estimate_pta.createOrReplaceTempView("after_educ_estimate_pta")

# COMMAND ----------

# DBTITLE 1,Calculate Difference in Estimate and Actual
# Calculate Difference in Estimate and Actual PtB
  
before_educ_diff_ptb = spark.sql("\
          Select clm_blg_prvdr_npi_num\
                ,clm_sbmt_chrg_amt\
                ,prediction\
                ,clm_sbmt_chrg_amt - prediction as incorrect_billing_amt\
           from before_educ_estimate_ptb")
  
  
after_educ_diff_ptb = spark.sql("\
          Select clm_blg_prvdr_npi_num\
          ,clm_sbmt_chrg_amt\
          ,prediction\
          ,clm_sbmt_chrg_amt - prediction as incorrect_billing_amt\
          from after_educ_estimate_ptb")
  
# Calculate Difference in Estimate and Actual DME
  
before_educ_diff_dme = spark.sql("\
          Select clm_blg_prvdr_npi_num\
                ,clm_sbmt_chrg_amt\
                ,prediction\
                ,clm_sbmt_chrg_amt - prediction as incorrect_billing_amt\
          from before_educ_estimate_dme")
  
  
after_educ_diff_dme = spark.sql("\
          Select clm_blg_prvdr_npi_num\
                ,clm_sbmt_chrg_amt\
                ,prediction\
                ,clm_sbmt_chrg_amt - prediction as incorrect_billing_amt\
          from after_educ_estimate_dme")
  
  
# Calculate Difference in Estimate and Actual Part A

before_educ_diff_pta = spark.sql("\
    Select clm_blg_prvdr_npi_num\
          ,clm_sbmt_chrg_amt\
          ,prediction\
          ,clm_sbmt_chrg_amt - prediction as incorrect_billing_amt\
    from before_educ_estimate_pta")
  
  
after_educ_diff_pta = spark.sql("\
    Select clm_blg_prvdr_npi_num\
          ,clm_sbmt_chrg_amt\
          ,prediction\
          ,clm_sbmt_chrg_amt - prediction as incorrect_billing_amt\
          from after_educ_estimate_pta")
  
  
  
  
# Union DataFrames of PartB and DME Together
before_educ_diff_dmeptb = before_educ_diff_ptb.union(before_educ_diff_dme)
before_educ_diff = before_educ_diff_dmeptb.union(before_educ_diff_pta)
after_educ_diff_dmeptb = after_educ_diff_ptb.union(after_educ_diff_dme)
after_educ_diff = after_educ_diff_dmeptb.union(after_educ_diff_pta)
  
  
  
# Find average, stdev of incorrect billing  and the number of claims (sample size)
before_agg = before_educ_diff.groupby("clm_blg_prvdr_npi_num").agg(avg("incorrect_billing_amt").alias("avg_incorrect_before")\
                                                                 ,stddev("incorrect_billing_amt").alias("stdev_incorrect_before")\
                                                                 ,count("clm_sbmt_chrg_amt").alias("sample_size_before"))

after_agg = after_educ_diff.groupby("clm_blg_prvdr_npi_num").agg(avg("incorrect_billing_amt").alias("avg_incorrect_after")\
                                                               ,stddev("incorrect_billing_amt").alias("stdev_incorrect_after")\
                                                               ,count("clm_sbmt_chrg_amt").alias("sample_size_after"))

# Forming Testing Data
  
  
testing_data = before_agg.join(after_agg , on = ['clm_blg_prvdr_npi_num'], how ='inner')
# engine.saveAsTable(testing_data,"{}.pe90_testing_data_all".format(options.ml_db))


# COMMAND ----------

testing_data.createOrReplaceTempView("testing_data")

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TEMPORARY VIEW filter_data AS (
# MAGIC SELECT     CAST(clm_blg_prvdr_npi_num as integer)
# MAGIC            ,avg_incorrect_before
# MAGIC            ,stdev_incorrect_before
# MAGIC            ,sample_size_before
# MAGIC            ,avg_incorrect_after
# MAGIC            ,stdev_incorrect_after
# MAGIC            ,sample_size_after
# MAGIC     FROM testing_data
# MAGIC     WHERE sample_size_before >= 50 AND
# MAGIC           sample_size_after >= 50    AND
# MAGIC           stdev_incorrect_before > 0 AND
# MAGIC     stdev_incorrect_after > 0);

# COMMAND ----------

# MAGIC %python
# MAGIC filtered_data = spark.table('filter_data')
# MAGIC filtered_data.persist()

# COMMAND ----------

# MAGIC %python
# MAGIC 
# MAGIC filtered_data = filtered_data.select('*').collect()

# COMMAND ----------

def conduct_Ttest(testingdata_collect):
  
  final_data = []

  for i in range(0,len(testingdata_collect)):
    
    before_mean = testingdata_collect[i]['avg_incorrect_before']
    before_stdev = testingdata_collect[i]['stdev_incorrect_before']
    before_samp = testingdata_collect[i]['sample_size_before']
    after_mean = testingdata_collect[i]['avg_incorrect_after']
    after_stdev = testingdata_collect[i]['stdev_incorrect_after']
    after_samp = testingdata_collect[i]['sample_size_after']
    t2, p2 = ttest_ind_from_stats(mean1 = before_mean, std1 = before_stdev, nobs1 = before_samp, mean2= after_mean, std2 = after_stdev, nobs2= after_samp, equal_var=False)
    
    #If incorrect greater after education or p-value is too big, flag; If p-value less than threshold, don't flag
    if (p2 < 0.05):
      flag = 'N'
    else:
      flag = 'Y'

    final_data.extend(((testingdata_collect[i]['clm_blg_prvdr_npi_num'], (after_mean - before_mean), float(t2), float(p2), flag),))

  schema = StructType([StructField("Educated_NPI", IntegerType())\
                          ,StructField("Averge_Difference", FloatType())\
                          ,StructField("Test_Value", FloatType())\
                          ,StructField("P_value", FloatType())\
                          ,StructField("Flag", StringType())])
    
  result_df =  spark.createDataFrame(final_data,schema=schema)
  result_df.write.format('delta').mode('overwrite').save('dbfs:/ml/pe90_model/result')


# COMMAND ----------

conduct_Ttest(filtered_data)
result_df = spark.read.format("delta").load('dbfs:/ml/pe90_model/result')

# COMMAND ----------

# MAGIC %sql
# MAGIC DROP TABLE IF EXISTS fps_mlasr.result

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE TABLE fps_mlasr.RESULT
# MAGIC USING DELTA
# MAGIC LOCATION 'dbfs:/ml/pe90_model/result';

# COMMAND ----------

pe90_testing_data_bef.createOrReplaceTempView('pe90_testing_data_before')
pe90_testing_data_aft.createOrReplaceTempView('pe90_testing_data_after')
pe90_testing_bef_parta.createOrReplaceTempView('pe90_testing_before_parta')
pe90_testing_aft_parta.createOrReplaceTempView('pe90_testing_after_parta')
pe90_most_curr_edu.createOrReplaceTempView('pe90_most_current_education')
fps_asrpt.createOrReplaceTempView('fps_asrpt')
fps_asrpt_alert_asctn.createOrReplaceTempView('fps_asrpt_alert_asctn')
fps_alert.createOrReplaceTempView('fps_alert')
fps_model.createOrReplaceTempView('fps_model')

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC CREATE OR REPLACE TEMPORARY VIEW skippedClaims AS (
# MAGIC     
# MAGIC 	Select beforemodel.clm_blg_prvdr_npi_num
# MAGIC           ,'B' as clm_src_type
# MAGIC           ,current_date as pe90_date
# MAGIC           ,'before' as before_after_educ
# MAGIC           ,beforemodel.data_count as before
# MAGIC           ,aftermodel.data_count as after
# MAGIC       from
# MAGIC           (Select clm_blg_prvdr_npi_num
# MAGIC                   ,count(clm_blg_prvdr_npi_num) as data_count
# MAGIC            from pe90_testing_data_before 
# MAGIC            where clm_src_type = 'B'
# MAGIC            group by clm_blg_prvdr_npi_num) as beforemodel
# MAGIC            
# MAGIC            left join 
# MAGIC            
# MAGIC            (Select clm_blg_prvdr_npi_num
# MAGIC                    ,count(clm_blg_prvdr_npi_num) as data_count
# MAGIC             from before_educ_estimate_ptb 
# MAGIC             group by clm_blg_prvdr_npi_num) as aftermodel
# MAGIC     
# MAGIC       on beforemodel.clm_blg_prvdr_npi_num = aftermodel.clm_blg_prvdr_npi_num 
# MAGIC       where beforemodel.data_count != aftermodel.data_count
# MAGIC             
# MAGIC 	UNION ALL
# MAGIC     
# MAGIC 	Select beforemodel.clm_blg_prvdr_npi_num
# MAGIC           ,'B' as clm_src_type
# MAGIC           ,current_date as pe90_date
# MAGIC           ,'after' as before_after_educ
# MAGIC           ,beforemodel.data_count as before
# MAGIC           ,aftermodel.data_count as after 
# MAGIC     from
# MAGIC           (Select clm_blg_prvdr_npi_num
# MAGIC                   ,count(clm_blg_prvdr_npi_num) as data_count
# MAGIC            from pe90_testing_data_after 
# MAGIC            where clm_src_type = 'B'
# MAGIC            group by clm_blg_prvdr_npi_num) as beforemodel
# MAGIC           
# MAGIC           left join 
# MAGIC           
# MAGIC           (Select clm_blg_prvdr_npi_num
# MAGIC                     ,count(clm_blg_prvdr_npi_num) as data_count
# MAGIC            from after_educ_estimate_ptb 
# MAGIC            group by clm_blg_prvdr_npi_num) as aftermodel
# MAGIC           
# MAGIC       on beforemodel.clm_blg_prvdr_npi_num = aftermodel.clm_blg_prvdr_npi_num 
# MAGIC       where beforemodel.data_count != aftermodel.data_count
# MAGIC           
# MAGIC 	UNION ALL
# MAGIC     
# MAGIC 	Select beforemodel.clm_blg_prvdr_npi_num
# MAGIC            ,'DME' as clm_src_type
# MAGIC            ,current_date as pe90_date
# MAGIC            ,'before' as before_after_educ
# MAGIC            ,beforemodel.data_count as before
# MAGIC            ,aftermodel.data_count as after 
# MAGIC     from
# MAGIC           (Select clm_blg_prvdr_npi_num
# MAGIC                   ,count(clm_blg_prvdr_npi_num) as data_count
# MAGIC            from pe90_testing_data_before 
# MAGIC            where clm_src_type = 'DME'
# MAGIC            group by clm_blg_prvdr_npi_num) as beforemodel
# MAGIC           
# MAGIC           left join 
# MAGIC           
# MAGIC           (Select clm_blg_prvdr_npi_num
# MAGIC                   ,count(clm_blg_prvdr_npi_num) as data_count
# MAGIC            from before_educ_estimate_dme
# MAGIC            group by clm_blg_prvdr_npi_num) as aftermodel
# MAGIC      on beforemodel.clm_blg_prvdr_npi_num = aftermodel.clm_blg_prvdr_npi_num 
# MAGIC      where beforemodel.data_count != aftermodel.data_count
# MAGIC             
# MAGIC 	UNION ALL
# MAGIC     
# MAGIC     Select beforemodel.clm_blg_prvdr_npi_num
# MAGIC           ,'DME' as clm_src_type
# MAGIC           ,current_date as pe90_date
# MAGIC           ,'after' as before_after_educ
# MAGIC           ,beforemodel.data_count as before
# MAGIC           ,aftermodel.data_count as after 
# MAGIC     from
# MAGIC           (Select clm_blg_prvdr_npi_num
# MAGIC                   , count(clm_blg_prvdr_npi_num) as data_count
# MAGIC            from pe90_testing_data_after
# MAGIC             where clm_src_type = 'DME'
# MAGIC             group by clm_blg_prvdr_npi_num) as beforemodel
# MAGIC             
# MAGIC             left join 
# MAGIC             
# MAGIC            (Select clm_blg_prvdr_npi_num
# MAGIC                    ,count(clm_blg_prvdr_npi_num) as data_count
# MAGIC             from after_educ_estimate_dme
# MAGIC             group by clm_blg_prvdr_npi_num) as aftermodel
# MAGIC             
# MAGIC 	on beforemodel.clm_blg_prvdr_npi_num = aftermodel.clm_blg_prvdr_npi_num 
# MAGIC 	where beforemodel.data_count != aftermodel.data_count
# MAGIC     
# MAGIC 	UNION ALL
# MAGIC     
# MAGIC 	Select beforemodel.clm_blg_prvdr_npi_num
# MAGIC            ,'A' as clm_src_type
# MAGIC            ,current_date as pe90_date
# MAGIC            ,'before' as before_after_educ
# MAGIC            ,beforemodel.data_count as before
# MAGIC            ,aftermodel.data_count as after 
# MAGIC      from
# MAGIC           (Select clm_blg_prvdr_npi_num
# MAGIC                   ,count(clm_blg_prvdr_npi_num) as data_count
# MAGIC            from pe90_testing_before_parta
# MAGIC            group by clm_blg_prvdr_npi_num) as beforemodel
# MAGIC            
# MAGIC            left join
# MAGIC            
# MAGIC            (Select clm_blg_prvdr_npi_num
# MAGIC                    ,count(clm_blg_prvdr_npi_num) as data_count
# MAGIC             from before_educ_estimate_pta
# MAGIC             group by clm_blg_prvdr_npi_num) as aftermodel
# MAGIC             
# MAGIC 	on beforemodel.clm_blg_prvdr_npi_num = aftermodel.clm_blg_prvdr_npi_num 
# MAGIC 	where beforemodel.data_count != aftermodel.data_count
# MAGIC     
# MAGIC 	UNION ALL
# MAGIC     
# MAGIC 	Select beforemodel.clm_blg_prvdr_npi_num
# MAGIC            ,'A' as clm_src_type
# MAGIC            ,current_date as pe90_date
# MAGIC            ,'after' as before_after_educ
# MAGIC            ,beforemodel.data_count as before
# MAGIC            ,aftermodel.data_count as after 
# MAGIC      from
# MAGIC           (Select clm_blg_prvdr_npi_num
# MAGIC                   ,count(clm_blg_prvdr_npi_num) as data_count
# MAGIC             from pe90_testing_after_parta
# MAGIC             group by clm_blg_prvdr_npi_num) as beforemodel
# MAGIC             
# MAGIC             left join 
# MAGIC             
# MAGIC           (Select clm_blg_prvdr_npi_num
# MAGIC                   ,count(clm_blg_prvdr_npi_num) as data_count
# MAGIC            from after_educ_estimate_pta
# MAGIC            group by clm_blg_prvdr_npi_num) as aftermodel
# MAGIC 	on beforemodel.clm_blg_prvdr_npi_num = aftermodel.clm_blg_prvdr_npi_num 
# MAGIC 	where beforemodel.data_count != aftermodel.data_count);

# COMMAND ----------

skipped = spark.table('skippedClaims')
skipped.cache()

# COMMAND ----------

skipped.write.format('delta').mode('overwrite').option("overwriteSchema", "true").save('dbfs:/ml/pe90_model/skipped')

# COMMAND ----------

# MAGIC %sql
# MAGIC DROP TABLE IF EXISTS fps_mlasr.skippedClaims

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE TABLE fps_mlasr.skippedClaims
# MAGIC USING DELTA
# MAGIC LOCATION 'dbfs:/ml/pe90_model/skipped';

# COMMAND ----------

(skipped.write.format("snowflake")
 .options(**sf_connection)
 .option("dbtable", "SKIPPED")
 .mode('append')
 .save())

# COMMAND ----------

result_df.createOrReplaceTempView("pe90_testing_results")
pe90_educatedlist.createOrReplaceTempView("pe90_educatedlist")
pe90_alledu_maxale.createOrReplaceTempView("pe90_alledu_maxale")
pe90_asrlist.createOrReplaceTempView("pe90_asrlist")

# COMMAND ----------

# DBTITLE 1,Education Data
# MAGIC %sql
# MAGIC CREATE OR REPLACE TEMPORARY VIEW education_data AS (
# MAGIC SELECT alleduc.asrpt_id
# MAGIC       ,CASE
# MAGIC           WHEN result.Flag IS NOT null THEN result.Flag
# MAGIC           WHEN result.Flag IS null AND usededuc.asrpt_id IS NOT null THEN 'Untestable'
# MAGIC           WHEN result.Flag IS null AND usededuc.asrpt_id IS null AND datediff(current_date, alleduc.max_alert_date) <= 365 THEN '<90days'
# MAGIC           ELSE 'N/A'
# MAGIC           END AS testing_flag
# MAGIC FROM pe90_alledu_maxale as alleduc
# MAGIC LEFT JOIN
# MAGIC pe90_educatedlist AS usededuc
# MAGIC ON usededuc.asrpt_id = alleduc.asrpt_id
# MAGIC LEFT JOIN
# MAGIC pe90_testing_results AS result
# MAGIC ON alleduc.subj_id = result.Educated_NPI); 

# COMMAND ----------

education_data_N = spark.sql("""Select * from education_data where testing_flag!='N' """)
education_data_N.createOrReplaceTempView("education_data_N") 

# COMMAND ----------

edu_data = spark.table('education_data')
edu_data.write.format('delta').mode('overwrite').save('dbfs:/ml/pe90_model/education_data')

# COMMAND ----------

# MAGIC %sql
# MAGIC DROP TABLE IF EXISTS fps_mlasr.education_data;

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE TABLE fps_mlasr.education_data
# MAGIC USING DELTA
# MAGIC LOCATION 'dbfs:/ml/pe90_model/education_data';

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC CREATE OR REPLACE TEMPORARY VIEW asr_education_date AS (
# MAGIC SELECT educ.asrpt_id
# MAGIC       ,educ.current_education_date
# MAGIC       ,result.Flag
# MAGIC       ,asr.subj_id
# MAGIC FROM pe90_most_current_education educ
# MAGIC INNER JOIN 
# MAGIC fps_asrpt AS asr 
# MAGIC ON asr.asrpt_id = educ.asrpt_id
# MAGIC INNER JOIN 
# MAGIC --pe90_testing_results_M AS result
# MAGIC (SELECT Educated_NPI
# MAGIC         ,Flag
# MAGIC  FROM pe90_testing_results
# MAGIC  WHERE  Flag = 'N') AS result
# MAGIC ON asr.subj_id = result.Educated_NPI);

# COMMAND ----------

asr_date = spark.table('asr_education_date')
asr_date.write.format('delta').mode('overwrite').save('dbfs:/ml/pe90_model/asr_education_date')

# COMMAND ----------

# MAGIC %sql
# MAGIC DROP TABLE IF EXISTS fps_mlasr.asr_education_date;

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE TABLE fps_mlasr.asr_education_date
# MAGIC USING DELTA
# MAGIC LOCATION 'dbfs:/ml/pe90_model/asr_education_date';

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TEMPORARY VIEW asr_education_date1 AS ( 
# MAGIC SELECT educ.asrpt_id
# MAGIC       ,educ.current_education_date
# MAGIC       ,educ.Flag
# MAGIC       ,educ.subj_id
# MAGIC       ,c.alert_creat_dt
# MAGIC       ,c.model_id
# MAGIC       ,c.alert_id
# MAGIC FROM asr_education_date AS educ
# MAGIC INNER JOIN
# MAGIC fps_asrpt_alert_asctn AS b 
# MAGIC ON educ.asrpt_id = b.asrpt_id
# MAGIC INNER JOIN
# MAGIC fps_alert AS c 
# MAGIC ON b.alert_id=c.alert_id 
# MAGIC INNER JOIN 
# MAGIC fps_model AS m 
# MAGIC ON c.model_id = m.model_id
# MAGIC WHERE m.MODEL_ACTVTY_STUS in ( 'Active' ,'Inactive')
# MAGIC GROUP BY educ.asrpt_id
# MAGIC         ,educ.current_education_date
# MAGIC         ,educ.Flag
# MAGIC         ,educ.subj_id
# MAGIC         ,c.alert_creat_dt
# MAGIC         ,c.model_id
# MAGIC         ,c.alert_id);

# COMMAND ----------

asr_date1 = spark.table('asr_education_date1')
asr_date1.write.format('delta').mode('overwrite').save('dbfs:/ml/pe90_model/asr_education_date1')

# COMMAND ----------

# MAGIC %sql
# MAGIC DROP TABLE IF EXISTS fps_mlasr.asr_education_date1;

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE TABLE fps_mlasr.asr_education_date1
# MAGIC USING DELTA
# MAGIC LOCATION 'dbfs:/ml/pe90_model/asr_education_date1';

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TEMPORARY VIEW alert_b AS (
# MAGIC 
# MAGIC SELECT alleduc.asrpt_id 
# MAGIC       ,alleduc.current_education_date 
# MAGIC       ,alleduc.Flag
# MAGIC       ,alleduc.subj_id
# MAGIC       ,alleduc.model_id
# MAGIC       ,alleduc.alert_creat_dt 
# MAGIC       ,alleduc.alert_id 
# MAGIC FROM asr_education_date1 alleduc
# MAGIC WHERE alleduc.alert_creat_dt < alleduc.current_education_date);

# COMMAND ----------

alert_b = spark.table('alert_b')
alert_b.write.format('delta').mode('overwrite').save('dbfs:/ml/pe90_model/alert_b')

# COMMAND ----------

# MAGIC %sql
# MAGIC DROP TABLE IF EXISTS fps_mlasr.alert_b;

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE TABLE fps_mlasr.alert_b
# MAGIC USING DELTA
# MAGIC LOCATION 'dbfs:/ml/pe90_model/alert_b';

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TEMPORARY VIEW alert_a AS (
# MAGIC 
# MAGIC SELECT alleduc.asrpt_id 
# MAGIC       ,alleduc.current_education_date 
# MAGIC       ,alleduc.Flag
# MAGIC       ,alleduc.subj_id
# MAGIC       ,alleduc.model_id
# MAGIC       ,alleduc.alert_creat_dt
# MAGIC       ,alleduc.alert_id   
# MAGIC FROM asr_education_date1 alleduc
# MAGIC WHERE alleduc.alert_creat_dt > alleduc.current_education_date);

# COMMAND ----------

alert_a = spark.table('alert_a')
alert_a.write.format('delta').mode('overwrite').save('dbfs:/ml/pe90_model/alert_a')

# COMMAND ----------

# MAGIC %sql
# MAGIC DROP TABLE IF EXISTS fps_mlasr.alert_a;

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE TABLE fps_mlasr.alert_a
# MAGIC USING DELTA
# MAGIC LOCATION 'dbfs:/ml/pe90_model/alert_a';

# COMMAND ----------

# MAGIC 
# MAGIC %sql
# MAGIC CREATE OR REPLACE TEMPORARY VIEW inal_data4_DS AS (
# MAGIC SELECT DISTINCT a.asrpt_id 
# MAGIC                 ,'New Model' AS testing_flag 
# MAGIC FROM alert_a AS a 
# MAGIC LEFT JOIN 
# MAGIC alert_b AS b 
# MAGIC ON a.asrpt_id = b.asrpt_id 
# MAGIC WHERE a.model_id NOT IN (select b.model_id 
# MAGIC                         FROM alert_b b 
# MAGIC                         WHERE  b.asrpt_id = a.asrpt_id) 
# MAGIC GROUP BY a.asrpt_id, a.subj_id, a.current_education_date 
# MAGIC ORDER BY a.asrpt_id);

# COMMAND ----------

inal_data4_DS = spark.table('inal_data4_DS')
inal_data4_DS.write.format('delta').mode('overwrite').save('dbfs:/ml/pe90_model/inal_data4_DS')

# COMMAND ----------

# MAGIC %sql
# MAGIC DROP TABLE IF EXISTS fps_mlasr.inal_data4_DS;

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE TABLE fps_mlasr.inal_data4_DS
# MAGIC USING DELTA
# MAGIC LOCATION 'dbfs:/ml/pe90_model/inal_data4_DS';

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TEMPORARY VIEW final_data_5_DS AS (
# MAGIC 
# MAGIC 
# MAGIC SELECT DISTINCT a.asrpt_id 
# MAGIC       ,'N' AS testing_flag 
# MAGIC FROM 
# MAGIC (SELECT asrpt_id 
# MAGIC  FROM alert_a 
# MAGIC  UNION 
# MAGIC  SELECT asrpt_id 
# MAGIC  FROM alert_b) a 
# MAGIC WHERE a.asrpt_id NOT IN (SELECT b.asrpt_id 
# MAGIC                          FROM inal_data4_DS  b)
# MAGIC GROUP BY a.asrpt_id
# MAGIC ORDER BY a.asrpt_id);

# COMMAND ----------

final_data_5_DS = spark.table('final_data_5_DS')
final_data_5_DS.write.format('delta').mode('overwrite').save('dbfs:/ml/pe90_model/final_data_5_DS')

# COMMAND ----------

# MAGIC %sql
# MAGIC DROP TABLE IF EXISTS fps_mlasr.final_data_5_DS;

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE TABLE fps_mlasr.final_data_5_DS
# MAGIC USING DELTA
# MAGIC LOCATION 'dbfs:/ml/pe90_model/final_data_5_DS';

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC CREATE OR REPLACE TEMPORARY VIEW After_education_flag AS (
# MAGIC SELECT * 
# MAGIC FROM inal_data4_DS
# MAGIC UNION ALL 
# MAGIC SELECT * 
# MAGIC FROM final_data_5_DS
# MAGIC UNION ALL
# MAGIC SELECT * 
# MAGIC FROM education_data_N);

# COMMAND ----------

After_education_flag = spark.table('After_education_flag')
After_education_flag.write.format('delta').mode('overwrite').save('dbfs:/ml/pe90_model/After_education_flag')

# COMMAND ----------

# MAGIC %sql
# MAGIC DROP TABLE IF EXISTS fps_mlasr.After_education_flag;

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE TABLE fps_mlasr.After_education_flag
# MAGIC USING DELTA
# MAGIC LOCATION 'dbfs:/ml/pe90_model/After_education_flag';

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC CREATE OR REPLACE TEMPORARY VIEW pe90_incr AS (
# MAGIC SELECT asr.asrpt_id
# MAGIC       ,current_date as pe90_date
# MAGIC       ,Case 
# MAGIC           WHEN educ.testing_flag IS null THEN 'N/A' 
# MAGIC           ELSE educ.testing_flag 
# MAGIC           END AS pe90_flag
# MAGIC FROM pe90_asrlist as asr
# MAGIC LEFT JOIN
# MAGIC After_education_flag AS educ
# MAGIC ON asr.asrpt_id = educ.asrpt_id);

# COMMAND ----------

pe90_incr = spark.table('pe90_incr')
pe90_incr.persist()
pe90_incr.write.format('delta').mode('overwrite').save('dbfs:/ml/pe90_model/pe90_incr')

# COMMAND ----------

# MAGIC %sql
# MAGIC DROP TABLE IF EXISTS fps_mlasr.pe90_incr

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE TABLE fps_mlasr.pe90_incr
# MAGIC USING DELTA
# MAGIC LOCATION 'dbfs:/ml/pe90_model/pe90_incr';

# COMMAND ----------

(pe90_incr.write.format("snowflake")
 .options(**sf_connection)
 .option("dbtable", "PE90_INCR")
 .mode('append')
 .save())
