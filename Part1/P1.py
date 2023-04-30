from __future__ import print_function
import sys
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
import pyspark.sql.types as T
from pyspark.ml.feature import Tokenizer, HashingTF, IDF
from pyspark.ml.classification import LogisticRegression

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: sort <file>", file=sys.stderr)
        sys.exit(-1)

    spark = SparkSession\
        .builder\
        .appName('Toxic Comment Classification')\
        .getOrCreate()
    train = spark.read\
            .option("multiLine", "true")\
            .option("header", "true")\
            .option("inferSchema", "true") \
            .option("quote", "\"") \
            .option("escape", "\"") \
            .csv("/P2/Part1/input/train.csv")
    test = spark.read\
            .option("multiLine", "true")\
            .option("header", "true")\
            .option("inferSchema", "true") \
            .option("quote", "\"") \
            .option("escape", "\"") \
            .csv("/P2/Part1/input/test.csv")
    df_train=train.na.fill("")
    df_test=test.na.fill("")
    out_cols = [i for i in df_train.columns if i not in ["id", "comment_text"]]

    df_train.filter(F.col('toxic') == 1)
    tokenizer = Tokenizer(inputCol="comment_text", outputCol="words")
    wordsData = tokenizer.transform(df_train)
    hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures")
    tf = hashingTF.transform(wordsData)
    tf.select('rawFeatures')
    idf = IDF(inputCol="rawFeatures", outputCol="features")
    idfModel = idf.fit(tf) 
    tfidf = idfModel.transform(tf)
    tfidf.select("features")
    lr = LogisticRegression(featuresCol="features", labelCol='toxic', regParam=0.1)
    lrModel = lr.fit(tfidf.limit(5000))
    res_train = lrModel.transform(tfidf)
    res_train.select("id", "toxic", "probability", "prediction")
    extract_prob = F.udf(lambda x: float(x[1]), T.FloatType())
    res_train.withColumn("proba", extract_prob("probability")).select("proba", "prediction")

    test_tokens = tokenizer.transform(df_test)
    test_tf = hashingTF.transform(test_tokens)
    test_tfidf = idfModel.transform(test_tf)
    test_res = df_test.select('id')
    test_probs = []
    for col in out_cols:
        lr = LogisticRegression(featuresCol="features", labelCol=col, regParam=0.1)
        lrModel = lr.fit(tfidf)
        res = lrModel.transform(test_tfidf)
        test_res = test_res.join(res.select('id', 'probability'), on="id")
        test_res = test_res.withColumn(col, extract_prob('probability')).drop("probability")
    
    test_res.write.option("header",True).csv("hdfs://10.128.0.6:9000/P2/Part1/output/result.csv")
    test_res.show(5)

    spark.stop()
