package sparkfm

import org.apache.spark.{SparkConf}
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.{Spark3StringIndexerUtils}
import org.apache.spark.sql.functions.{col, lit, concat}
import org.elasticsearch.spark.sql._

object IndexModelES {

  lazy val conf = new SparkConf()
    .setAppName("spark_fm")


  val spark = SparkSession.builder()
    .config(conf)
    //.master("local[*]")
    .getOrCreate()

  import spark.implicits._

  // Saved from previous stage of training. Should come from a config file.
  val featureOffsets = Map("user_id_index" -> 0,
    "age_index" -> 9923,
    "title_index" -> 6040,
    "occupation_index" -> 9932,
    "gender_index" -> 9930)


  def main(args: Array[String]): Unit = {

    val ml1mPath = args(0)
    val indexersPath = args(1)
    val modelPath = args(2)

    val model = spark.read.option("header","true").parquet(modelPath)
    model.show(4)

    val movies = readCsv(ml1mPath + "/movies.dat")
      .toDF("movie_id","title","genres")
    val titleIndexer = Spark3StringIndexerUtils.load(indexersPath + "/title", spark)
    val moviesIndexed = titleIndexer
      .transform(movies)
      .withColumn("title_index",$"title_index"+lit(featureOffsets("title_index")))

    // index movies to es
    val movieDocs = moviesIndexed
      .join(model,$"title_index"===$"index")
      .withColumn("feature_type",lit("movie"))
      .withColumn("id",concat(lit("movie_"),$"index"))
      .select("id","feature_type","embedding","bias","title")
    movieDocs.show(4)

    movieDocs.saveToEs("recsys",Map("es.mapping.id" -> "id"))

    // index user features docs
    val users = readCsv(ml1mPath + "users.dat")
      .toDF("user_id","gender","age","occupation","zipcode")

    // Read pipeline
    // When elasticksearch-spark can be run on spark 3.0.0 just use Pipline.load()
    val userFeaturesIndexers = Array("user_id","gender","age","occupation")
      .map(indexersPath + "/"+_)
      .map(Spark3StringIndexerUtils.load(_,spark))
    val pipeline = Spark3StringIndexerUtils.pipelineModelConstruct(userFeaturesIndexers)
    val usersIndexed = pipeline
      .transform(users)
      .select(Array("user_id_index","gender_index","age_index","occupation_index")
        .map(name=>(col(name) + lit(featureOffsets(name))).alias(name)):_*)


    val userfeatureColumns = Seq("user_id","age","gender","occupation")
    userfeatureColumns.foreach(columnName =>
      usersIndexed
        .dropDuplicates(columnName+"_index")
        .join(model, col(columnName+"_index")===$"index")
        .withColumn("feature_type",lit(columnName))
        .withColumn("id",concat(lit(columnName+"_"),$"index"))
        .select("id","feature_type","embedding","bias")
        .saveToEs("recsys",Map("es.mapping.id" -> "id")))

    spark.close()
  }

  // Used to read csv with :: delimiter
  def readCsv(path:String) = {
    spark.sqlContext.read
      .option("inferSchema", "true")
      .option("delimiter", "\t")
      .csv(spark.sqlContext.read.textFile(path)
        .map(line => line.split("::").mkString("\t")))
  }

}
