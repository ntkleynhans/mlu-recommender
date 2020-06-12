from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql.functions import desc
from pyspark.mllib.feature import HashingTF, IDF

conf = SparkConf().setMaster("local").setAppName("SparkTFIDF")
sc = SparkContext(conf=conf)
spark = SparkSession(sc)

rawData = sc.textFile("../data/subset-small.tsv")
fields = rawData.map(lambda x: x.split("\t"))
documents = fields.map(lambda x: x[3].split(" "))

documentNames = fields.map(lambda x: x[1])

hashingTF = HashingTF(100000)
tf = hashingTF.transform(documents)

tf.cache()
idf = IDF(minDocFreq=2).fit(tf)
tfidf = idf.transform(tf)

gettysburgTF = hashingTF.transform(['Gettysburg'])
gettysburgHashValue = int(gettysburgTF.indices[0])

gettsburgRelevance = tfidf.map(lambda x: float(x[gettysburgHashValue]))

zippedResults = gettsburgRelevance.zip(documentNames)

schema = StructType([StructField("score", FloatType(), True), StructField("document", StringType(), True)])

resultSchema = spark.createDataFrame(zippedResults, schema)
resultSchema.createOrReplaceTempView('Results')

print("Result: ")
print(resultSchema.sort(desc('score')).show())

