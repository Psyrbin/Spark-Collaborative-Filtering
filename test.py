import sys
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.mllib.evaluation import RankingMetrics
from pyspark.sql.functions import collect_list, explode

seed = 0

memory='10g'
spark = SparkSession.builder.appName('rs').config('spark.executor.memory', memory) \
             .config('spark.driver.memory', memory) \
             .config('spark.executor.memoryOverhead', memory) \
             .config("spark.sql.broadcastTimeout", "36000") \
             .config("spark.storage.memoryFraction","0") \
             .config("spark.memory.offHeap.enabled","true") \
             .config("spark.memory.offHeap.size",memory).getOrCreate()

# argv[1] = 0 if need to train a model, 1 if already trained

test = spark.read.parquet(sys.argv[2])

if sys.argv[1]:
    model = ALSModel.load(sys.argv[3])
else:
    train = spark.read.parquet(sys.argv[1])
    als = ALS(rank=sys.argv[3], regParam=sys.argv[4], seed=seed)
    model = als.fit(train.toDF('user', 'item', 'rating'))

        

predictions_test = model.transform(test.toDF('user', 'item', 'rating'))
evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
rmse = evaluator.evaluate(predictions_test)

print("Root-mean-square error = " + str(rmse))


test.createOrReplaceTempView('test')
test_true = spark.sql('select user, book from test where rating > 2 sort by rating desc')
labels = test_true.groupby('user').agg(collect_list('book'))

test_recommendations = model.recommendForUserSubset(labels.select('user'), 500)
preds = test_recommendations.withColumn('recommendations', explode('recommendations')).select('user', 'recommendations.item').groupBy('user').agg(collect_list('item'))

preds_and_labels = preds.join(labels, on='user')


metrics = RankingMetrics(preds_and_labels.select('collect_list(item)', 'collect_list(book)').rdd)
map_metric = metrics.meanAveragePrecision                                                      
pA = metrics.precisionAt(500)                                                            
ndcgA = metrics.ndcgAt(500)

results.append((rank, reg, rmse, map_metric, pA, ndcgA))

print('MAP = ', map_metric, ' pA = ', pA, ' ndcgA = ', ndcgA, '\n')

res_rdd = spark.sparkContext.parallelize(results)
res_df = spark.createDataFrame(res_rdd).repartition(1)
res_df.write.csv('test_results.csv')