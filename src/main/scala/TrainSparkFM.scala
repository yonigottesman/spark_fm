
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




object TrainSparkFM {

  lazy val conf = new SparkConf()
    .setAppName("spark_fm")


  val spark = SparkSession.builder()
    .config(conf)
    .master("local[*]")
    .getOrCreate()

  import spark.implicits._
  def main(args: Array[String]): Unit = {



    val movies = spark.read.option("delimiter","::")
      .csv("ml-1m/movies.dat")
      .toDF("movie_id","title","genres")

    val titleIndexer = new StringIndexer()
      .setInputCol("title")
      .setOutputCol("title_index")
      .fit(movies)

    val moviesIndexed = titleIndexer.transform(movies)

    titleIndexer.write.overwrite().save("indexers/title")

    val users = spark.read.option("delimiter","::")
      .csv("ml-1m/users.dat")
      .toDF("user_id","gender","age","occupation","zipcode")


    val userFeaturesIndexers = Array("user_id","gender","age","occupation")
      .map(col => new StringIndexer().setInputCol(col).setOutputCol(col+"_index"))
    val pipe = new Pipeline().setStages(userFeaturesIndexers).fit(users)

    val usersIndexed = pipe.transform(users)
    pipe.write.overwrite().save("indexers/user_features")

    val ratings = spark.read.option("delimiter","::")
      .csv("ml-1m/ratings.dat")
      .toDF("user_id","movie_id","rating","time")
      .withColumn("rating",$"rating".cast(IntegerType))

    val ratingsJoined = ratings
      .join(usersIndexed,Seq("user_id"))
      .join(moviesIndexed,Seq("movie_id"))

    ratingsJoined.show(10)


    // movies feature sizes
    val numMovies = moviesIndexed
      .select(countDistinct($"title_index").alias("numMovies"))
      .take(1)(0).getAs[Long]("numMovies")

    // users feature sizes
    val userfeatureColumns = Seq("user_id_index","age_index","gender_index","occupation_index")
    val userFeaturSizesDf = usersIndexed
      .select(userfeatureColumns.map(c => countDistinct(col(c)).alias(c)): _*)
    val userFeatureSizes = Map(userfeatureColumns
      .zip(userFeaturSizesDf.take(1)(0).toSeq.map(_.asInstanceOf[Long])):_*)

    //create offset map
    val featureColumns = Seq("user_id_index","title_index","age_index","gender_index","occupation_index")
    val featureSizes = userFeatureSizes + ("title_index"->numMovies)
    val featureOffsets = Map(featureColumns.zip(featureColumns.scanLeft(0L)((agg,current)=>agg+featureSizes(current)).dropRight(1)):_*)



    // add offset to each column
    val ratingsInput = ratingsJoined
      .select(featureColumns.map(name=>(col(name) + lit(featureOffsets(name))).alias(name)):+$"rating":_*)



    val featureVectorSize = featureSizes.values.sum

    //convert rows to sparse vector
    val data = ratingsInput
      .withColumn("features",createFeatureVectorUdf(lit(featureVectorSize)+:featureColumns.map(col):_*))

    val Array(trainingData, testData) = data.randomSplit(Array(0.9, 0.1))


    val fm = new FMRegressor()
      .setLabelCol("rating")
      .setFeaturesCol("features")
      .setFactorSize(150)
      .setStepSize(0.01)

    trainingData.select("rating","features").show(10,false)
    trainingData.printSchema()
    val model = fm.fit(trainingData)

    val predictions = model.transform(testData)

    val evaluator = new RegressionEvaluator()
      .setLabelCol("rating")
      .setPredictionCol("prediction")
      .setMetricName("rmse")
    val rmse = evaluator.evaluate(predictions)

    println(s"Root Mean Squared Error (RMSE) on test data = $rmse")


    model.write.overwrite().save("fmmodel")

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
