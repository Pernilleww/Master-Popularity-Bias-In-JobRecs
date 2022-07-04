import sys

import os.path
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from scipy import spatial
import tensorflow as tf
import numpy as np
import math
from xQuAD.Model import Model

#taken from github: https://github.com/biasinrecsys/wsdm2021

def compute_weights(item_id, tail_head, tail_long, mrtype, list_u, p_d_u):
    psum = 0
    for tset_pos, tset in enumerate([tail_head, tail_long]):
        p_v_d = (1 if item_id in tset else 0)
        ies = ((np.prod([1 - (1 if ranked_item_id in tset else 0) for ranked_item_id in list_u])) if mrtype == 'balanced' else (1 - len(list(set(tset) & set(list_u))) / len(list_u))) if len(list_u) > 1 else 1
        psum += p_d_u[tset_pos] * p_v_d * ies
    return psum

class RankerXQuad(Model):

    def __init__(self, users, items, observed_relevance, unobserved_relevance, item_field, user_field, rating_field, item_feature_df):
        super().__init__(users, items, observed_relevance, unobserved_relevance, item_field, user_field, rating_field, item_feature_df)

    def rerank(self, type='smooth', lmbda=0.4, k=10, rmax=100):
        #tail_head = np.argsort(self.item_popularity / np.sum(self.item_popularity))[::-1][:head_tail_split]
        #tail_long = np.argsort(self.item_popularity / np.sum(self.item_popularity))[::-1][head_tail_split:]
        assert len(self.tail_head) + len(self.tail_long) == self.no_items
        print(lmbda)

        for user_id, user_observed in zip(self.users, self.observed_relevance):
            print('\rPerforming reranking for user', user_id, '/', self.no_users, end='')
            user_scores = self.predicted_relevance[user_id]
            if sum(user_scores) == 0:
                continue

            #user_scores = self.predicted_relevance[self.predicted_relevance.userId == user_id]
            #user_scores['score'] = (user_scores.score.values - min(user_scores.score.values)) / (max(user_scores.score.values) - min(user_scores.score.values))
            user_scores = (user_scores - min(user_scores)) / (max(user_scores) - min(user_scores))

            train_pids = np.nonzero(user_observed)[0]
            user_scores[train_pids] = -10000
            #user_scores[user_scores.itemId.isin(train_pids)] = -1000
            list_u = []
            p_d_u = [np.sum(user_observed[self.tail_head]) / np.sum(user_observed), np.sum(user_observed[self.tail_long]) / np.sum(user_observed)]
            
            assert np.sum(p_d_u) == 1
            #self.predicted_relevance[self.predicted_relevance.userId == user_id].score = np.zeros(self.no_items)
            self.predicted_relevance[user_id] = np.zeros(self.no_items)
            most_relevant_items = np.argsort(-user_scores)[:rmax]

            #most_relevant_items = np.argsort(-user_scores.scores.values)[:rmax]
            while len(list_u) < k:
                kwargs = {"tail_head": self.tail_head, "tail_long": self.tail_long, "mrtype": type, "list_u": list_u, "p_d_u": p_d_u}
                weights = np.apply_along_axis(compute_weights, 0, np.expand_dims(most_relevant_items, axis=0), **kwargs)
                comb_scores = np.array([(1 - lmbda) * user_scores[item_id] + lmbda * weights[item_pos] for item_pos, item_id in enumerate(most_relevant_items)])
                list_u.append(most_relevant_items[np.argsort(comb_scores)[-1]])
                self.predicted_relevance[user_id, list_u[-1]] = (k - len(list_u)) / k
                most_relevant_items = np.delete(most_relevant_items, np.argsort(comb_scores)[-1])
            
