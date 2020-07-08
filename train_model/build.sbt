name := "sparkfm_train"

version := "0.1"

scalaVersion := "2.12.10"


lazy val spark = "3.0.0"

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % spark,
  "org.apache.spark" %% "spark-sql" % spark,
  "org.apache.spark" %% "spark-streaming" % spark,
  "org.apache.spark" %% "spark-mllib" % spark
)

