name := "sparkfm_index"

version := "0.1"

scalaVersion := "2.11.12"

lazy val spark = "2.4.6"

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % spark,
  "org.apache.spark" %% "spark-sql" % spark,
  "org.apache.spark" %% "spark-streaming" % spark,
  "org.apache.spark" %% "spark-mllib" % spark
)

libraryDependencies += "org.elasticsearch" %% "elasticsearch-spark-20" % "7.8.0"

