~/spark-3.0.0-bin-hadoop2.7/bin/spark-submit \
--executor-memory 10G \
--driver-memory=8g \
--class sparkfm.TrainSparkFM \
target/scala-2.12/sparkfm_train_2.12-0.1.jar \
ml-1m/