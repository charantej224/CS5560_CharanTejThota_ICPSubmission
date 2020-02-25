from pyspark.ml.feature import HashingTF, IDF, Tokenizer
from pyspark.sql import SparkSession, Window
from pyspark.sql.functions import monotonically_increasing_id
import pyspark.sql.functions as F

spark = SparkSession.builder.appName("TfIdf-tokenizing").getOrCreate()
documents = spark.read.text("dataset/*.txt")
#documents = documents.withColumn("doc_id", monotonically_increasing_id())
documents = documents.withColumn("doc_id", F.row_number().over(Window.orderBy('value')))

documents.printSchema()
# creating tokens/words from the sentence data
tokenizer = Tokenizer(inputCol="value", outputCol="words")
wordsData = tokenizer.transform(documents)

# applying tf on the words data
hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures", numFeatures=20)
featurizedData = hashingTF.transform(wordsData)
# alternatively, CountVectorizer can also be used to get term frequency vectors

# calculating the IDF
idf = IDF(inputCol="rawFeatures", outputCol="features")
idfModel = idf.fit(featurizedData)
rescaledData = idfModel.transform(featurizedData)

# displaying the results
rescaledData.select("doc_id", "features").show()

# closing the spark session
spark.stop()
