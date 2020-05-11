import sys
from pyspark.sql import SparkSession
from pyspark.sql.types import IntegerType
from pyspark.sql.functions import ntile
from pyspark.sql import Window

seed = 0

spark = SparkSession.builder.getOrCreate()

users = spark.read.parquet(sys.argv[1])
interactions = spark.read.parquet(sys.argv[2])


interactions = interactions.toDF('user', 'book', 'is_read', 'rating', 'is_reviewed')
interactions = interactions.select('user', 'book', 'rating')
interactions = interactions.withColumn('user', interactions['user'].cast(IntegerType()))
interactions = interactions.withColumn('book', interactions['book'].cast(IntegerType()))
interactions = interactions.withColumn('rating', interactions['rating'].cast(IntegerType()))


user_splits = users.randomSplit([0.6, 0.2, 0.2], seed=seed)
user_splits[0].createOrReplaceTempView('train_users')
user_splits[1].createOrReplaceTempView('val_users')
user_splits[2].createOrReplaceTempView('test_users')

interactions.createOrReplaceTempView('interactions')

byUser = Window.partitionBy('user').orderBy('book')
val_interactions_raw = spark.sql('SELECT * FROM interactions WHERE user IN (SELECT _c0 FROM val_users)')
val_interactions_raw = val_interactions_raw.select('user', 'book', 'rating', ntile(2).over(byUser).alias('half'))

test_interactions_raw = spark.sql('SELECT * FROM interactions WHERE user IN (SELECT _c0 FROM test_users)')
test_interactions_raw = test_interactions_raw.select('user', 'book', 'rating', ntile(2).over(byUser).alias('half'))

val_interactions_raw.createOrReplaceTempView('val_raw')
test_interactions_raw.createOrReplaceTempView('test_raw')


val = spark.sql('SELECT user, book, rating FROM val_raw where half = 1')
val_train = spark.sql('SELECT user, book, rating FROM val_raw where half = 2')

test = spark.sql('SELECT user, book, rating FROM test_raw where half = 1')
test_train = spark.sql('SELECT user, book, rating FROM test_raw where half = 2')


train = spark.sql('SELECT * FROM interactions WHERE user IN (SELECT _c0 FROM train_users)').union(val_train).union(test_train)

val.createOrReplaceTempView('val')
train.createOrReplaceTempView('train')
test.createOrReplaceTempView('test')

val = spark.sql('SELECT * FROM val WHERE book IN (SELECT DISTINCT(book) FROM train)')
test = spark.sql('SELECT * FROM test WHERE book IN (SELECT DISTINCT(book) FROM train)')

train.write.parquet(sys.argv[3], '_train.parquet')
val.write.parquet(sys.argv[3], '_val.parquet')
test.write.parquet(sys.argv[3], '_test.parquet')