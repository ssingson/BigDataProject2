#!/bin/bash
source ../../env.sh
../../start.sh
hadoop dfsadmin -safemode leave
/usr/local/hadoop/bin/hdfs dfs -rm -r /P2/Part3/input/
/usr/local/hadoop/bin/hdfs dfs -rm -r /P2/Part3/output/
/usr/local/hadoop/bin/hdfs dfs -mkdir -p /P2/Part3/input/
/usr/local/hadoop/bin/hdfs dfs -copyFromLocal train.csv /P2/Part3/input/
/usr/local/hadoop/bin/hdfs dfs -copyFromLocal test.csv /P2/Part3/input/
/usr/local/spark/bin/spark-submit --master=spark://$SPARK_MASTER:7077 Q3.py hdfs://$SPARK_MASTER:9000/P2/Part1/input/
../../stop.sh
