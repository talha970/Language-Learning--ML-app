/**
  * Created by pradyumnad on 10/07/15.
  */

import java.io.{BufferedWriter, File, FileWriter}
import java.nio.file.{Files, Paths}

import org.apache.spark.mllib.clustering.{KMeans, KMeansModel}
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.model.RandomForestModel
import org.apache.spark.rdd.RDD
import org.apache.spark.streaming.{Seconds, StreamingContext}
import org.apache.spark.{SparkConf, SparkContext}
import org.openimaj.feature.local.list.LocalFeatureList
import org.openimaj.image.feature.local.engine.DoGSIFTEngine
import org.openimaj.image.{ImageUtilities, MBFImage}
import org.openimaj.image.feature.local.keypoints.Keypoint


object IPApp {
  val IMAGE_CATEGORIES = List("cars", "planes", "ships" )

  /**
    * @note Test method for classification on Spark
    * @param sc : Spark Context
    * @return
    */
  def testImageClassification(sc: SparkContext, path: String): String ={

    val model = KMeansModel.load(sc, IPSettings.KMEANS_PATH)
    val vocabulary = ImageUtils.vectorsToMat(model.clusterCenters)
    val desc = ImageUtils.bowDescriptors(path, vocabulary)
    val histogram = ImageUtils.matToVector(desc)

    println("-- Histogram size : " + histogram.size)
    println(histogram.toArray.mkString(" "))

    val nbModel = RandomForestModel.load(sc, IPSettings.RANDOM_FOREST_PATH)
    val p = nbModel.predict(histogram)
    (IMAGE_CATEGORIES(p.toInt))
  }

  def testImageClassificationF(sc:SparkContext,path:String) :String={
    // Features Extraction
    var maxp : Int = 0;
    val outputFolder = "output/";
    val IMAGE_CATEGORIES = List("cars", "planes", "ships" )
    val mbfImage: MBFImage = ImageUtilities.readMBF(new File(path))
    val doGSIFTEngine: DoGSIFTEngine = new DoGSIFTEngine
    val features: LocalFeatureList[Keypoint] = doGSIFTEngine.findFeatures(mbfImage.flatten)
    val fw: FileWriter = new FileWriter(outputFolder + "features" + ".txt")
    val bw: BufferedWriter = new BufferedWriter(fw)
    var i: Int = 0
    while (i < features.size) {
      {
        val c: Array[Double] = features.get(i).getFeatureVector.asDoubleVector
        bw.write(0 + ",")
        var j: Int = 0
        while (j < c.length) {
          {
            bw.write(c(j) + " ")
          }
          {
            j += 1; j - 1
          }
        }
        bw.newLine()
      }
      {
        i += 1; i - 1
      }
    }
    bw.close()

    val test = sc.textFile("features.txt")
    val nbModel = RandomForestModel.load(sc, IPSettings.RANDOM_FOREST_PATH)

    val testData1 = test.map(line => {
      val parts = line.split(',')
      LabeledPoint(parts(0).toDouble, Vectors.dense(parts(1).split(' ').map(_.toDouble)))
    })

    val classify1 = testData1.map { line =>
      val prediction = nbModel.predict(line.features)
      (line.label, prediction)
    }

    val prediction1 = classify1.groupBy(_._1).map(f => {
      var fuzzy_Pred = Array(0, 0, 0)
      f._2.foreach(ff => {
        fuzzy_Pred(ff._2.toInt) += 1
      })
      var count = 0.0
      fuzzy_Pred.foreach(f => {
        count += f
      })
      var i = -1
      var maxIndex = 3
      val max = fuzzy_Pred.max
      val pp = fuzzy_Pred.map(f => {
        val p = f * 100 / count
        i = i + 1
        if(f == max){
          maxIndex=i
          maxp = i
        }

        (i, p)
      })
      (f._1, pp, maxIndex)
    })
    prediction1.foreach(f => {
      println("\n\n\n" + f._1 + " : " + f._2.mkString(";\n"))
    })
    val y: RDD[(Double, Double)] = prediction1.map(f => {
      (f._3.toDouble,f._1 )
    })

    (IMAGE_CATEGORIES(maxp))

  }


  def testImage(string: String):String = {
    val conf = new SparkConf()
      .setAppName(s"IPApp")
      .setMaster("local[*]")
      .set("spark.executor.memory", "6g")
      .set("spark.driver.memory", "6g")

    val sparkConf = new SparkConf().setAppName("SparkWordCount").setMaster("local[*]")
    val sc= SparkContext.getOrCreate(sparkConf)
    val res = testImageClassification(sc, string)

    printf(res);
    res
  }
}