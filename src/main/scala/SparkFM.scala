
import org.apache.spark.SparkConf
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.regression.{FMRegressionModel, FMRegressor}
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.sql.functions.countDistinct
import org.apache.spark.sql.functions.{col, lit, udf}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.types.IntegerType




object SparkFM {

  lazy val conf = new SparkConf()
    .setAppName("spark_fm")


  val spark = SparkSession.builder()
    .config(conf)
    .master("local[*]")
    .getOrCreate()

  import spark.implicits._
  def main(args: Array[String]): Unit = {



    val movies = spark.read.option("delimiter","::")
      .csv("/Users/yonatang/recommendation_playground/ml-1m/movies.dat")
      .toDF("movie_id","title","genres")

    val users = spark.read.option("delimiter","::")
      .csv("/Users/yonatang/recommendation_playground/ml-1m/users.dat")
      .toDF("user_id","gender","age","occupation","zipcode")


    val ratings = spark.read.option("delimiter","::")
      .csv("/Users/yonatang/recommendation_playground/ml-1m/ratings.dat")
      .toDF("user_id","movie_id","rating","time")
      .withColumn("rating",$"rating".cast(IntegerType))


    val ratingsJoined = ratings
      .join(users,Seq("user_id"))
      .join(movies,Seq("movie_id"))



    val indexers = Array("user_id","gender","age","occupation","title")
      .map(col => new StringIndexer().setInputCol(col).setOutputCol(col+"_index"))
    val pipe = new Pipeline().setStages(indexers)
    val ratingsIndexed = pipe.fit(ratingsJoined).transform(ratingsJoined)

    val featureColumns = Seq("user_id_index","title_index","age_index","gender_index","occupation_index")

    val featureSizesDf = ratingsIndexed.select(featureColumns.map(c => countDistinct(col(c)).alias(c)): _*)
    val featureSizes = Map(featureColumns.zip(featureSizesDf.take(1)(0).toSeq.map(_.asInstanceOf[Long])):_*)
    val featureOffsets = Map(featureColumns.zip(featureColumns.scanLeft(0L)((agg,current)=>agg+featureSizes(current)).dropRight(1)):_*)



    val ratings_input = ratingsIndexed
      .select(featureColumns.map(name=>(col(name) + lit(featureOffsets(name))).alias(name)):+$"rating":_*)



    val featureVectorSize = featureSizes.values.sum
    val data = ratings_input
      .withColumn("features",createFeatureVectorUdf(lit(featureVectorSize)+:featureColumns.map(col):_*))

    val Array(trainingData, testData) = data.randomSplit(Array(0.9, 0.1))


    val fm = new FMRegressor()
      .setLabelCol("rating")
      .setFeaturesCol("features")
      .setFactorSize(150)
      .setStepSize(0.001)

    val model = fm.fit(trainingData)

    val predictions = model.transform(testData)

    // Select example rows to display.
    predictions.select("prediction", "rating", "features").show(5)


    val evaluator = new RegressionEvaluator()
      .setLabelCol("rating")
      .setPredictionCol("prediction")
      .setMetricName("rmse")
    val rmse = evaluator.evaluate(predictions)
    println(s"Root Mean Squared Error (RMSE) on test data = $rmse")


    model.save("fmmodel")

    spark.close()

  }

  val createFeatureVectorUdf = udf((size:Int,
                                user_id_index:Int,
                                movie_index:Int,
                                age_index:Int,
                                gender_index:Int,
                                occupation_index:Int) =>
    Vectors.sparse(size,Array(user_id_index,movie_index,age_index,gender_index,occupation_index),Array.fill(5)(1)))


}
