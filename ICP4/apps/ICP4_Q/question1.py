import pyspark
from pyspark.sql import SQLContext

sc = pyspark.SparkContext()
sqlContext = SQLContext(sc)

rdd = sqlContext.read.csv("data.csv", header=True, inferSchema=True)

gender_filter = ['Female']

# 1 Filter is transformation and Count is action
print("----------------------------------------------------------------------------------")
print("1 Filter is transformation and Count is action")
print("----------------------------------------------------------------------------------")
print(rdd.filter(rdd.gender != 'Female').count())

# 2 Distinct is transformation and show is the action
print("----------------------------------------------------------------------------------")
print("2 Distinct is transformation and show is the action")
print("----------------------------------------------------------------------------------")
return_values = rdd.distinct().show(5)

# 3 groupBy is transformation and count and collect are the action
print("----------------------------------------------------------------------------------")
print("3 groupBy is transformation and count and collect are the action")
print("----------------------------------------------------------------------------------")
print(rdd.groupBy(rdd.gender).count().collect())
