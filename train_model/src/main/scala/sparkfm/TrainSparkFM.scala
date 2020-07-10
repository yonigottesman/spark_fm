package sparkfm

import org.apache.spark.SparkConf
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.{StringIndexer, StringIndexerModel}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.regression.FMRegressor
import org.apache.spark.sql.{SaveMode, SparkSession}
import org.apache.spark.sql.functions.{col, countDistinct, lit, udf}
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

    val ml1mPath = args(0)

    val movies = spark.read.option("delimiter","::")
      .csv(ml1mPath+"/movies.dat")
      .toDF("movie_id","title","genres")

    val titleIndexer = new StringIndexer()
      .setInputCol("title")
      .setOutputCol("title_index")
      .fit(movies)

    val moviesIndexed = titleIndexer.transform(movies)

    titleIndexer.write.overwrite().save("indexers/title")

    val users = spark.read.option("delimiter","::")
      .csv(ml1mPath+"/users.dat")
      .toDF("user_id","gender","age","occupation","zipcode")

    val userFeatures = Array("user_id","gender","age","occupation")
    val userFeaturesIndexers = userFeatures
      .map(col => new StringIndexer().setInputCol(col).setOutputCol(col+"_index"))
    val pipe = new Pipeline().setStages(userFeaturesIndexers).fit(users)

    // save each indexer because the es indexing job is spark 2.4 and formats are not the same
    // once elasticsearch-spark supports scala 2.12 just save pipeline
    userFeatures.zip(pipe.stages)
      .foreach{case (featureName, stringIndexer:StringIndexerModel) =>
        stringIndexer.write.overwrite().save("indexers/"+featureName)}

    val usersIndexed = pipe.transform(users)

    val ratings = spark.read.option("delimiter","::")
      .csv(ml1mPath+"/ratings.dat")
      .toDF("user_id","movie_id","rating","time")
      .withColumn("rating",$"rating".cast(IntegerType))

    val ratingsJoined = ratings
      .join(usersIndexed,Seq("user_id"))
      .join(moviesIndexed,Seq("movie_id"))


    usersIndexed.dropDuplicates("gender").show(10,false)
    usersIndexed.dropDuplicates("age").show(10,false)


    val featureColumns = Seq("user_id_index","title_index","age_index","gender_index","occupation_index")

    ratingsJoined.select((featureColumns:+"rating").map(col):_*).show(4)


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
    val featureSizes = userFeatureSizes + ("title_index"->numMovies)
    val featureOffsets = Map(featureColumns
      .zip(featureColumns.scanLeft(0L)((agg,current)=>agg+featureSizes(current)).dropRight(1)):_*)

    println(featureOffsets)
    println(featureSizes)

    // add offset to each column
    val ratingsInput = ratingsJoined
      .select(featureColumns.map(name=>(col(name) + lit(featureOffsets(name))).alias(name)):+$"rating":_*)

    val featureVectorSize = featureSizes.values.sum

    //convert rows to sparse vector
    val data = ratingsInput
      .withColumn("features",createFeatureVectorUdf(lit(featureVectorSize)+:featureColumns.map(col):_*))


    data.select("features","rating").show(10,false)


    val Array(trainset, testset) = data.randomSplit(Array(0.9, 0.1))

// best
//    val fm = new FMRegressor()
//      .setLabelCol("rating")
//      .setFeaturesCol("features")
//      .setFactorSize(120)
//      .setMaxIter(300)
//      .setRegParam(0.01)
//      .setStepSize(0.01)



    val fm = new FMRegressor()
      .setLabelCol("rating")
      .setFeaturesCol("features")
      .setFactorSize(120)
      .setMaxIter(300)
      .setRegParam(0.01)
      .setStepSize(0.01)


    val model = fm.fit(trainset)

    val testPredictions = model.transform(testset)
    val trainPredictions = model.transform(trainset)

    val evaluator = new RegressionEvaluator()
      .setLabelCol("rating")
      .setPredictionCol("prediction")
      .setMetricName("rmse")



    val testRMSE = evaluator.evaluate(testPredictions)
    val trainRMSE = evaluator.evaluate(trainPredictions)

    println(s"test rmse = $testRMSE. train rms = $trainRMSE")


    model.write.overwrite().save("fmmodel")

    //save embeddings and biases in table format
    val matrixRows = model.factors.rowIter.toSeq.map(_.toArray).zip(model.linear.toArray)
      .zipWithIndex.map { case ((a, b), i) => (i, b, a) }

    spark.sparkContext
      .parallelize(matrixRows)
      .toDF("index","bias","embedding")
      .write.mode(SaveMode.Overwrite).option("header","true").parquet("model_raw")

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
