import pandas as pd
import numpy as np
from pyspark.mllib.evaluation import BinaryClassificationMetrics


# Average Recommendation Popularity
# L_u list of recommendations for user, 
# U_t is the number of users in test set
# This measure from calculates the average popularity of the recommended
# items in each list. For any given recommended item in the list,
# we measure the average number of ratings for those items
def ARP(train_items, test_users, recommendations, k=10, pyspark_ALS = False):

    rec_pop = 0
    for user in test_users:
        theta = 0
        if pyspark_ALS:
            for item in recommendations[recommendations.userId == user].itemId.values:
                theta += _theta(item, train_items)
        else:
            for item in recommendations[user]:
                theta += _theta(item, train_items)
        rec_pop += (theta / k)
    
    return rec_pop / len(test_users)

def _theta(i, train_items):
    return np.count_nonzero(train_items == i)

# Average Percentage of Long Tail Items (ALPT)
# Is used in[2], this metric measures the average percentage of long tail items in
# the recommended lists and it is defined as follows
# def APLT(long_tail_items, test_matrix, recommendations):
#     long_tail = 0
#     for user in test_matrix.userId.unique():
#         rec_long_tail_intersect = len(list(set(recommendations[recommendations.userId == user].itemId.to_numpy()).intersection(long_tail_items)))
#         rec_list_size  =   recommendations[recommendations.userId == user].shape[0]
#         if rec_list_size > 0:
#             long_tail +=  rec_long_tail_intersect / rec_list_size
#     return long_tail / test_matrix.userId.nunique()

def APLT(recommendations, test_users, long_tail_items, k=10, pyspark_ALS = False):
    long_tail = 0
    for user in test_users:
        if pyspark_ALS:
            rec_long_tail_intersect = len(list(set(recommendations[recommendations.userId == user].itemId.to_numpy()).intersection(long_tail_items)))
            rec_list_size  =   recommendations[recommendations.userId == user].shape[0]
            long_tail +=  rec_long_tail_intersect / k
        else:
            rec_user_at_k = recommendations[user]
            rec_items_in_long_tail = len(rec_user_at_k[np.in1d(rec_user_at_k, long_tail_items)])
            long_tail += rec_items_in_long_tail / k
    return long_tail/ len(np.unique(test_users))


# Average Coverage of Long Tail items
# ACLT measures what fraction of the long-tail items the recommender has
# covered:
# def ACLT(long_tail_items, test_matrix, recommendations):
#     total_indic = 0
#     for user in test_matrix.userId.unique():
#         indic = [1 if item in long_tail_items else 0 for item in recommendations[recommendations.userId == user].itemId]
#         total_indic += sum(indic)
#         print(indic, total_indic)
#     return total_indic / test_matrix.userId.nunique()

def ACLT(recommendations, test_users, long_tail_items, k=10, pyspark_ALS = False):
    total_indic = 0
    for user in test_users:
        if pyspark_ALS:
            indic = [1 if item in long_tail_items else 0 for item in recommendations[recommendations.userId == user].itemId]
            total_indic += sum(indic)
        else:
            y_pred = recommendations[user]
            top_k = np.argsort(-y_pred)[:k]
            indic = [1 if item in long_tail_items else 0 for item in y_pred]
            total_indic += sum(indic)
    
    print(indic,  len(np.unique(test_users)))
    return total_indic / len(np.unique(test_users))



def GAP(group, rating_matrix, user_profile):
    theta = 0
    rec_pop = 0
    for user in group:
        for item in user_profile[user_profile.userId == user].itemId:
            theta += matrix[matrix.itemId ==  item].shape[0]
        rec_pop += theta / user_profile.shape[0]
    gap = rec_pop / len(group)
    return gap

def delta_GAP(group):
    return "TODO"




def RMSE(error, num):
    return np.sqrt(error / num)


def MAE(error_mae, num):
    return (error_mae / num)




# Scala version implements .roc() and .pr()
# Python: https://spark.apache.org/docs/latest/api/python/_modules/pyspark/mllib/common.html
# Scala: https://spark.apache.org/docs/latest/api/java/org/apache/spark/mllib/evaluation/BinaryClassificationMetrics.html
class CurveMetrics(BinaryClassificationMetrics):
    def __init__(self, *args):
        super(CurveMetrics, self).__init__(*args)

    def _to_list(self, rdd):
        points = []
        # Note this collect could be inefficient for large datasets 
        # considering there may be one probability per datapoint (at most)
        # The Scala version takes a numBins parameter, 
        # but it doesn't seem possible to pass this from Python to Java
        for row in rdd.collect():
            # Results are returned as type scala.Tuple2, 
            # which doesn't appear to have a py4j mapping
            points += [(float(row._1()), float(row._2()))]
        return points

    def get_curve(self, method):
        rdd = getattr(self._java_model, method)().toJavaRDD()
        return self._to_list(rdd)

