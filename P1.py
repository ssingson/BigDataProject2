from __future__ import print_function
import sys
import pandas as pd
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
import pyspark.sql.types as T
from pyspark.ml.feature import Tokenizer, HashingTF, IDF
from pyspark.ml.classification import LogisticRegression

def to_spark_df(fin):
    """
    Parse a filepath to a spark dataframe using the pandas api.
    
    Parameters
    ----------
    fin : str
        The path to the file on the local filesystem that contains the csv data.
        
    Returns
    -------
    df : pyspark.sql.dataframe.DataFrame
        A spark DataFrame containing the parsed csv data.
    """
    df = pd.read_csv(fin)
    df.fillna("", inplace=True)
    df = hc.createDataFrame(df)
    return(df)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: sort <file>", file=sys.stderr)
        sys.exit(-1)

    spark = SparkSession\
        .builder\
        .appName('Toxic Comment Classification')\
        .enableHiveSupport()\
        .config("spark.executor.memory", "4G")\
        .config("spark.driver.memory","18G")\
        .config("spark.executor.cores","7")\
        .config("spark.python.worker.memory","4G")\
        .config("spark.driver.maxResultSize","0")\
        .config("spark.sql.crossJoin.enabled", "true")\
        .config("spark.serializer","org.apache.spark.serializer.KryoSerializer")\
        .config("spark.default.parallelism","2")\
        .getOrCreate()
    spark.sparkContext.setLogLevel('INFO')

    train = to_spark_df("/P2/Part1/input/train.csv")
    test = to_spark_df("/P2/Part1/input/test.csv")
    out_cols = [i for i in train.columns if i not in ["id", "comment_text"]]
    train.show(5)

