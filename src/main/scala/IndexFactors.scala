import org.apache.spark.SparkConf
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.regression.{FMRegressionModel, FMRegressor}
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.sql.functions.countDistinct
import org.apache.spark.sql.functions.{col, lit, udf}
import org.apache.spark.ml.linalg.{Matrix, Vectors}
import org.apache.spark.sql.types.IntegerType


object IndexFactors {
  lazy val conf = new SparkConf()
    .setAppName("spark_fm")


  val spark = SparkSession.builder()
    .config(conf)
    .master("local[*]")
    .getOrCreate()

  import spark.implicits._

  def main(args: Array[String]): Unit = {
    val fmModel:FMRegressionModel = FMRegressionModel.load("fmmodel")



    val matrixRows = fmModel.factors.rowIter.toSeq.map(_.toArray).zipWithIndex
    val df = spark.sparkContext.parallelize(matrixRows).toDF("factors","index")

    df.show(10)

//    println(s"Factors: ${fmModel.factors} Linear: ${fmModel.linear} " +
//      s"Intercept: ${fmModel.intercept}")
  }
}
