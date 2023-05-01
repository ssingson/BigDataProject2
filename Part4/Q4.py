from __future__ import print_function
import sys
import math

from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.functions import when, lit

from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, DecisionTreeClassifier
from pyspark.mllib.evaluation import MultilabelMetrics, BinaryClassificationMetrics
from pyspark.ml.linalg import Vector
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder, StandardScaler

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: sort <file>", file=sys.stderr) 
        sys.exit(-1)
    
    spark = SparkSession\
            .builder\
            .appName("IncomeClassification")\
            .getOrCreate()

    #get column names
    columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']

    #Pull training data
    df_train=spark.read\
            .format("csv")\
            .option("inferSchema","true")\
            .load("/P2/Part4/input/train.csv", columns = columns)

    #Clean data, have classifier column
    df_train = df_train.toDF(*columns)

    df_train = df_train.withColumn("grade", \
            when((df_train.income == ' <=50K'), 0) \
            .otherwise(1) \
            )

    #Pull test data
    df_test=spark.read\
            .format("csv")\
            .option("inferSchema","true")\
            .load("/P2/Part4/input/test.csv", columns = columns)

    #Clean data, have classifier column
    df_test = df_test.toDF(*columns)

    df_test = df_test.withColumn("grade", \
            when((df_test.income == ' <=50K'), 0) \
            .otherwise(1))

    #For categorical features, have them in indexed list form instead
    indexer = StringIndexer()\
            .setInputCols(['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country'])\
            .setOutputCols(['workclass_index', 'education_index', 'marital-status_index', 'occupation_index', 'relationship_index', 'race_index', 'sex_index', 'native-country_index'])

    #For categorical features
    encoder = OneHotEncoder()\
            .setInputCols(['workclass_index', 'education_index', 'marital-status_index', 'occupation_index', 'relationship_index', 'race_index', 'sex_index', 'native-country_index'])\
            .setOutputCols(['workclass_encoded', 'education_encoded', 'marital-status_encoded', 'occupation_encoded', 'relationship_encoded', 'race_encoded', 'sex_encoded', 'native-country_encoded'])

    #Have all features in one vector to match format of MLLib
    assembler = VectorAssembler()\
            .setInputCols(['age', 'workclass_encoded', 'fnlwgt', 'education_encoded', 'education-num', 'marital-status_encoded', 'occupation_encoded', 'relationship_encoded', 'race_encoded', 'sex_encoded', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country_encoded'])\
            .setOutputCol('vectorized_features')

    #Have features scaled based on standard deviations from the feature's average to avoid bias from varying ranges
    scaler = StandardScaler()\
            .setInputCol('vectorized_features')\
            .setOutputCol('features')

    #Pipeline the steps for data cleaning into one pipeline
    pipeline_stages = Pipeline()\
            .setStages([indexer, encoder, assembler, scaler])

    #Clean data in the training data, utilize same cleaning with the test data
    pipeline_model = pipeline_stages.fit(df_train)
    train = pipeline_model.transform(df_train)
    test = pipeline_model.transform(df_test)

    #Fit and transform the training and test data
    rf = RandomForestClassifier(featuresCol = 'features', labelCol = 'grade')
    rfModel = rf.fit(train)
    predictions = rfModel.transform(test)

    #Compute and print the accuracy
    accuracy = predictions.filter(predictions.grade == predictions.prediction).count() / float(predictions.count())
    print("The accuracy of the dataset using a Random Forest Classifier is %s." % accuracy)

    #Fit and transform the training and test data
    dt = DecisionTreeClassifier(featuresCol = 'features', labelCol = 'grade')
    dtModel = dt.fit(train)
    predictions = dtModel.transform(test)

    #Compute and print the accuracy
    accuracy = predictions.filter(predictions.grade == predictions.prediction).count() / float(predictions.count())
    print("The accuracy of the dataset using a Decision Tree Classifier is %s." % accuracy)

    spark.stop()
