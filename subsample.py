from pyspark.sql import SparkSession
import sys

spark = SparkSession.builder.getOrCreate()
seed = 0
users = spark.read.csv('hdfs:/user/bm106/pub/goodreads/user_id_map.csv')
subsample = users.sample(False, float(sys.argv[1]), seed=seed)
# users.createOrReplaceTempView('users')
interactions = spark.read.csv('hdfs:/user/bm106/pub/goodreads/goodreads_interactions.csv')
interactions.createOrReplaceTempView('interactions')
subsample.createOrReplaceTempView('users_sub')
interactions_sub = spark.sql('select interactions._c0, interactions._c1, interactions._c2, interactions._c3, interactions._c4 from interactions right join users_sub on interactions._c0 = users_sub._c0')
interactions_sub.createOrReplaceTempView('interactions_sub')  
interactions_sub_drop = spark.sql('SELECT * FROM interactions_sub WHERE _c0 IN (SELECT _c0 from interactions_sub WHERE _c3 > 0 GROUP BY _c0 HAVING COUNT(*) > 10)')
interactions_sub_drop.createOrReplaceTempView('isb')
subsample = spark.sql('SELECT * FROM users_sub WHERE _c0 in (SELECT _c0 FROM isb)')

subsample.write.parquet(sys.argv[1], '_users.parquet')
interactions_sub_drop.write.parquet(sys.argv[1], '_interactions.parquet')