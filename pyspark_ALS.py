

from data import get_interactions, get_job_data, get_rating_matrix, get_job_interactions_matrix

from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row
import pyspark.pandas as ps
from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql import functions as F
from pyspark.mllib.evaluation import MulticlassMetrics, RankingMetrics


from metrices import *

def pyspark_ALS(rating_matrix, n, long_tail_items, implicit=False):
    spark = SparkSession.builder.appName('Recommendation_system2').getOrCreate()    
    #ratings =spark.createDataFrame(rating_matrix)
    #ratings = spark.createDataFrame(ratingsRDD)
    #(train, test) = ratings.randomSplit([0.8, 0.2])

    train = rating_matrix.sample(frac=0.8 , random_state=99)
    test = rating_matrix.loc[~rating_matrix.index.isin(train.index), :]
    ts = rating_matrix.drop(columns=['rating'])

    test_rdd = spark.createDataFrame(test)
    train_rdd = spark.createDataFrame(train)
    print("Start training")

    # Build the recommendation model using ALS on the training data
    # Note we set cold start strategy to 'drop' to ensure we don't get NaN evaluation metrics
    als = ALS(maxIter=5, regParam=0.01, userCol="userId", itemCol="itemId", ratingCol="rating",
            coldStartStrategy="drop", implicitPrefs=implicit)
    
    
    # Add hyperparameters and their respective values to param_grid
    param_grid = ParamGridBuilder() \
                .addGrid(als.rank, [10, 50, 100, 150]) \
                .addGrid(als.regParam, [.01, .05, .1, .15]) \
                .build()

    # Define evaluator as RMSE and print length of evaluator
    evaluator = RegressionEvaluator(
           metricName="rmse", 
           labelCol="rating", 
           predictionCol="prediction") 
    print ("Num models to be tested: ", len(param_grid))

    # Build cross validation using CrossValidator
    cv = CrossValidator(estimator=als, estimatorParamMaps=param_grid, evaluator=evaluator, numFolds=5)
    
    #Fit cross validator to the 'train' dataset
    model = cv.fit(spark.createDataFrame(train))#Extract best model from the cv model above
    
    print("Trining Done")

    best_model = model.bestModel# View the predictions
    test_predictions = best_model.transform(test_rdd)

    print("Creating recommendations")
    # Generate top n movie recommendations for each user
    userRecs = best_model.recommendForUserSubset( test_rdd, 10)
    userRecs = userRecs\
        .withColumn("rec_exp", F.explode("recommendations"))\
        .select('userId', F.col("rec_exp.itemId"), F.col("rec_exp.rating"))

    # Generate top n user recommendations for each movie
    jobRecs = best_model.recommendForItemSubset(test_rdd, 10)
    jobRecs = jobRecs\
        .withColumn("rec_exp", F.explode("recommendations"))\
        .select('itemId', F.col("rec_exp.userId"), F.col("rec_exp.rating"))

    global_user = None
    user_list = np.array([], dtype=np.float64)
    testItems = list()
    true_list = test.userId
    for row in test.iterrows():

        if row[1]['userId'] != global_user:
            user_list = np.append(user_list, row[1]['userId'])
            testItems.append(int(row[1]['itemId']))
            global_user = row[1]['userId']
        else:
            testItems.append(int(row[1]['itemId']))
        true_list[global_user] = testItems

    pandasDf = pd.DataFrame({'userId': user_list})
    sub_user = spark.createDataFrame(pandasDf)
    labelsList = list()
    for user, items in best_model.recommendForUserSubset(sub_user, n).collect():
        predict_items = [i.itemId for i in items]
        labelsList.append((predict_items, true_list[user]))
    
    labels = spark.sparkContext.parallelize(labelsList)
    ranking_metrics = RankingMetrics(labels)
    binary_metrics = BinaryClassificationMetrics(labels)
    
    # Instantiate metrics object
    #predictionAndLabels = test.rdd.map(lambda lp: (float(best_model.predict(lp.features)), lp.label))
    #test.take(2)
    #metrics = MulticlassMetrics(predictionAndLabels)

    rmse = evaluator.evaluate(test_predictions)
    pandas_user_recs = userRecs.toPandas()

    arp = ARP(train.itemId.values, test.userId.unique(), recommendations=pandas_user_recs, pyspark_ALS=True, k=10 )
    aplt = APLT(long_tail_items = long_tail_items, test_users=test.userId.unique(), recommendations=pandas_user_recs, pyspark_ALS=True, k=10 )
    aclt = ACLT(long_tail_items = long_tail_items, test_users = test.userId.unique(), recommendations = pandas_user_recs, pyspark_ALS=True, k=10 )
    precision =  ranking_metrics.precisionAt(10)

    #f1Score = metrics.fMeasure()
    recall = ranking_metrics.recallAt(10)
    #area_under_roc = binary_metrics.areaUnderROC
    ndcg = ranking_metrics.ndcgAt(10)
    

    
    print("**Best Model**")# Print "Rank"
    print("  Rank:", best_model._java_obj.parent().getRank())# Print "MaxIter"
    print("  MaxIter:", best_model._java_obj.parent().getMaxIter())# Print "RegParam"
    print("  RegParam:", best_model._java_obj.parent().getRegParam())




    print("  Summary Stats")
    print("  RMSE test", rmse)
    print("  Precision = %s" % precision)
    print("  Recall = %s" % recall)
    print("  NDCG = %s" % ndcg)
    #print("  F1 Score = %s" % f1Score)
    #print("  Area under ROC = %s" % area_under_roc)


    print("  Summary Popularity Bias")
    print("  ARP@10", arp)
    print("  APLT@10", aplt)
    print("  ACLT@10", aclt)
    

    userRecs_n = best_model.recommendForAllUsers( n)
    userRecs_n = userRecs_n\
        .withColumn("rec_exp", F.explode("recommendations"))\
        .select('userId', F.col("rec_exp.itemId"), F.col("rec_exp.rating"))

    # Generate top n user recommendations for each movie
    # jobRecs_n = best_model.recommendForItemSubset(ratings, n)
    # jobRecs_n = jobRecs_n\
    #     .withColumn("rec_exp", F.explode("recommendations"))\
    #     .select('itemId', F.col("rec_exp.userId"), F.col("rec_exp.rating"))
    

    pandas_user_recs_n = userRecs_n.toPandas()

    
    return pandas_user_recs_n, test, train


