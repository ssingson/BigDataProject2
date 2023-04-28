#!/bin/bash
source ../../env.sh
../../start.sh
hadoop dfsadmin -safemode leave
/usr/local/hadoop/bin/hdfs dfs -rm -r /P2/Part1/input/
/usr/local/hadoop/bin/hdfs dfs -mkdir -p /P2/Part1/input/
/usr/local/hadoop/bin/hdfs dfs -copyFromLocal ../../p2data/train.csv /P2/Part1/input/
/usr/local/hadoop/bin/hdfs dfs -copyFromLocal ../../p2data/test.csv /P2/Part1/input/
/usr/local/spark/bin/spark-submit --master=spark://$SPARK_MASTER:7077 ./P1.py hdfs://$SPARK_MASTER:9000/q3/input/
../../stop.sh
