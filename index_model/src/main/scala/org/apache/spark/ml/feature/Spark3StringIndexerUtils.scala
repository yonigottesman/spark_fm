package org.apache.spark.ml.feature

import org.apache.hadoop.fs.Path
import org.apache.spark.ml.{Transformer,PipelineModel}
import org.apache.spark.ml.util.DefaultParamsReader
import org.apache.spark.sql.SparkSession
import org.glassfish.jersey.process.internal.Stages

object Spark3StringIndexerUtils {

  
  def load(path: String, spark:SparkSession): Transformer = {
    val metadata = DefaultParamsReader.loadMetadata(path, spark.sparkContext)
    val dataPath = new Path(path, "data").toString

    val data = spark.read.parquet(dataPath)
      .select("labelsArray")
      .head()
    val labelsArray = data.getAs[Seq[Seq[String]]](0).map(_.toArray).toArray

    val model = new StringIndexerModel(metadata.uid, labelsArray(0))
    metadata.getAndSetParams(model)
    model
  }

  def pipelineModelConstruct(stages:Array[Transformer]) = {
    new PipelineModel("pipe",stages)
  }

}
