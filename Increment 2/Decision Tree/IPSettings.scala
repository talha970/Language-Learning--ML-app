/**
  * Created by pradyumnad on 19/07/15.
  */
object IPSettings {

  val PATH = "data/"
//  val INPUT_DIR = PATH + "train"
  val INPUT_DIR = PATH + "dtrain"
//  val TEST_INPUT_DIR = PATH + "test2"
//  val TEST_INPUT_DIR = PATH + "dtrain"
  val TEST_INPUT_DIR = PATH + "dtest"
  val FEATURES_PATH = PATH + "model/features"

  val KMEANS_PATH = PATH + "model/clusters"
  val KMEANS_CENTERS_PATH = PATH + "model/clusterCenters"
  val HISTOGRAM_PATH = PATH + "model/histograms"
  val RANDOM_FOREST_PATH = PATH + "model/nbmodel"
  val DECISION_TREE_PATH = PATH + "model/dtmodel"
}
