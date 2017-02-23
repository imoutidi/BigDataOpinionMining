import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{HashingTF, StopWordsRemover, Tokenizer}
import org.apache.spark.sql.SQLContext
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.log4j.Logger
import org.apache.log4j.Level
import scala.io.Source

//case class for storing the train data for the model
case class MovieReviewTable(label: Double, text: String, fileName: String)
//case class for storing the test data for the model
case class ReviewTable(text: String)

object LRClassifier {

  def main(args: Array[String]) {

    Logger.getLogger("org").setLevel(Level.ERROR)
    Logger.getLogger("akka").setLevel(Level.ERROR)
    val conf = new SparkConf().setAppName("Labeled LR")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)
    import sqlContext.implicits._ // We need this in order to use Dataframes (.toDF())

    //The default Spark stopwords + some extra words that will be removed from the dataset e.g. Names, nouns.
    val cleanWords = Source.fromFile(args(3)).getLines.toArray.map(x => x.toLowerCase) 
    
    // SparkContext’s TextFile method, i.e., sc.textFile in Spark Shell, creates an RDD with each line as an element. 
    // If there are 10 files in the folder, 10 partitions will be created. For that reason we have to repartition our RDD.
    val negatives = sc.wholeTextFiles(args(0) + "/neg/").coalesce(sc.defaultParallelism)
    val possitives = sc.wholeTextFiles(args(0) + "/pos/").coalesce(sc.defaultParallelism)
        
    // we use some regular expresions for cleaning the data from numbers, special characters such as (,.":?) 
    // and words with one or two characters	
    val cleanNegs = negatives.map ({case (name, contents) =>  MovieReviewTable(0.0, contents.replaceAll("[^a-zA-Z ]", " ")
    	.replaceAll("\\b\\w{1,2}\\b", "").trim.toLowerCase, name.split("/").last)})
    val cleanPos = possitives.map ({case (name, contents) =>  MovieReviewTable(1.0, contents.replaceAll("[^a-zA-Z ]", " ")
    	.replaceAll("\\b\\w{1,2}\\b", "").trim.toLowerCase, name.split("/").last)})

    //Accessing elements
    //cleanNegs.foreach(x => println(x.text))
	
    //Merging the negative and possitive reviews into one dataframe
    val dataTrain = cleanNegs.union(cleanPos).toDF().coalesce(sc.defaultParallelism)

    /////////////////////////////////////////////////////////////////////////////////////////
    //Preparing the classifier
    val tokenizer = new Tokenizer().setInputCol("text").setOutputCol("words")

    val remover = new StopWordsRemover().setStopWords(cleanWords).setInputCol(tokenizer.getOutputCol)
    .setOutputCol("filteredWords")

    val hashingTF = new HashingTF()
      .setInputCol(remover.getOutputCol).setOutputCol("features").setNumFeatures(100000)

    val logReg = new LogisticRegression()
      .setRegParam(0.5).setElasticNetParam(0).setMaxIter(10)

    val pipeline = new Pipeline()
      .setStages(Array(tokenizer, remover, hashingTF, logReg))

    // Training the Logistic Regresion Model
    val logRegModel = pipeline.fit(dataTrain)

    //Retrieving the Test Data
    val testData = sc.wholeTextFiles(args(1)).coalesce(sc.defaultParallelism)
    //Cleaning the test data
    val cleanTest = testData.map({case (name, contents) => ReviewTable(contents.replaceAll("[^a-zA-Z ]", " ")
    	.replaceAll("\\b\\w{1,2}\\b", "").trim.toLowerCase + " " + name.split("/|\\.")(9)) }).toDF()

    //Calculating the opinions of the test dataset
    //logRegModel.transform(cleanTest).drop("words").drop("filteredWords").drop("features").drop("probability").drop("rawPrediction")
    // .rdd.saveAsTextFile(args(2))
    //Getting the results of the classification
    val results = logRegModel.transform(cleanTest).drop("words").drop("filteredWords")
    .drop("features").drop("probability").drop("rawPrediction").rdd
    //Formating the results in 00000 1 format and collecting them into the main cluster
    val collectedResults = results.map(x => x.toString).map(y => y.split(" ")last)
    .map(z => z.replaceAll("[],]", " ")).map(w => w.dropRight(3)).collect()//.map(x => x.saveAsTextFile(args(2)))
    //Sorting the ids of the results
    scala.util.Sorting.quickSort(collectedResults)
    //Saving the collected results into a text file
    collectedResults.map(x => scala.tools.nsc.io.File("opinionsResults").appendAll(x + "\n"))
   }
}
