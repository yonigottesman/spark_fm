spark-submit \
--executor-memory 10G \
--class sparkfm.IndexModelES \
--jars elasticsearch-spark-20_2.11-7.8.0.jar \
target/scala-2.11/sparkfm_index_2.11-0.1.jar \
ml-1m/ indexers/ model_raw/