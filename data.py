
import pandas as pd
import datetime
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate



params = {
    'lookback-user': 1,
    'lookback-content': 40,
    'num_of_occupation_cap': 3,
    'adtypes': [40, 43, 44], # This is the three job verticals. Heltid,deltid and leder-stillinger
    'single_var_emb_names': ['county', 'job_sector','state','job_duration', 'ad_type','id'],  #id, needs to be the last emb
    'multi_var_emb_names' : ['occupation', 'industry']
}




def get_job_data(sqlContext, lookback, indicator = False):
    from_date = datetime.date.today() - datetime.timedelta(lookback)
    now = datetime.date.today().strftime("%Y-%m-%d")
    from_date = from_date.strftime("%Y-%m-%d")

    # id, ad_type, vertical, job_sector, job_duration, created_time, state, industry, occupation_general, occupation_specialization, county
    q = f"SELECT  id, ad_type, vertical, job_sector, created_time, state, industry, county, published FROM ad_content ac LEFT JOIN post_code pc ON ac.post_code = pc.code WHERE vertical = 'JOB' AND published >= '{from_date}'"
    job_data = FINNHelper.contentdb(sqlContext, q)

    job_data = job_data.toPandas()

    job_data.industry = job_data.industry.apply(lambda x: str(x).split())


    job_data = job_data.rename(columns={"id": "itemId"})

    # id, ad_type, vertical, job_sector, job_duration, created_time, state, industry, occupation_general, occupation_specialization, county
    q = f"SELECT id, applicationDeadline FROM ad_content_search as ac LEFT JOIN post_code pc ON ac.post_code = pc.code WHERE vertical = 'JOB' AND published >= '{from_date}'"
    job_data_search = FINNHelper.contentdb(sqlContext, q)
    job_data_search = job_data_search.toPandas()
    job_data_search = job_data_search.rename(columns={"id": "itemId"})

    job_data = job_data_search.merge(job_data, on=["itemId"], how="right", indicator=indicator)

    return job_data



def get_interactions(lake, lookback):

    raw = lake.filter(Func.col("location") == "job-browse").select(['itemId', 'userId',  'applyLoweffect', 'applyForPosition', 'click', "favorite", 'itemPos', 'timestamp']).filter((Func.col("userType").isin(['L', 'A'])))

    print('Loading events from now --> {} days back'.format(lookback))


    interactions = DATAHelper.toPandas(raw)

    print("Done processing event-data")

    interaction_events = interactions
    interaction_events = interactions.drop(columns=['itemPos'])
    interaction_events = interaction_events.drop_duplicates()
    favorite_event = interaction_events.sort_values(by='timestamp').query('favorite == 1').groupby(['userId', 'itemId'])['favorite', 'timestamp', 'click'].first().reset_index().rename(columns={'favorite': 'favorite_event'})
    applyForPosition_event = interaction_events.sort_values(by='timestamp').query('applyForPosition == 1').groupby(['userId', 'itemId'])['applyForPosition', 'timestamp', 'click'].first().reset_index().rename(columns={'applyForPosition': 'applyForPosition_event'})
    applyForPosition_event['applyForPosition'] = 1
    favorite_event['favorite'] = 1
    interaction_events = interaction_events.merge(applyForPosition_event, on=['userId', 'timestamp', 'applyForPosition', 'itemId', 'click'], how='left')
    interaction_events = interaction_events.merge(favorite_event, on=['userId', 'timestamp', 'favorite', 'itemId', 'click'], how='left')
    interaction_events.favorite_event = interaction_events.favorite_event.replace(np.nan, 0)
    interaction_events.applyForPosition_event = interaction_events.applyForPosition_event.replace(np.nan, 0)

    return interaction_events, interactions

def get_job_interactions_matrix(job_data, interactions, indicator = False):
    interactions_with_jobs = job_data.merge(interactions, on='itemId', how='left', indicator=indicator)
    recommendations = interactions.itemId.value_counts().rename_axis('itemId').reset_index(name='recommendations')

    job_interactions_matrix = (interactions_with_jobs.groupby(
    interactions_with_jobs['itemId']).agg(
        {
        'click': 'sum',
        'favorite_event': 'sum',
        'applyForPosition_event': 'sum',
        'county': 'first',
        'industry': 'first',
        'job_sector': 'first',
        'state': 'first',
        'applicationdeadline': 'first',
        'published': 'first'
        }
        ).reset_index())
    job_interactions_matrix = job_interactions_matrix.merge(recommendations, on='itemId', how='left')
    job_interactions_matrix['amount_deadline_days'] = (job_interactions_matrix['applicationdeadline'] - job_interactions_matrix['published']).dt.days
    job_interactions_matrix['amount_deadline_days'] = job_interactions_matrix.amount_deadline_days.replace(np.nan, 0)
    #Should maybe be removed
    #job_click_matrix = job_click_matrix[job_click_matrix["state"].str.contains("DEACTIVATED")==False]
    job_interactions_matrix['interactions'] = job_interactions_matrix[['click', 'favorite_event', 'applyForPosition_event']].sum(axis=1)
    job_interactions_matrix.recommendations = job_interactions_matrix.recommendations.replace(np.nan, 0)

    return job_interactions_matrix



def give_rating_based_on_interaction(row):
    if row['click'] == 1 and row['favorite_event'] == 0 and row['applyForPosition_event'] == 0:
        return 1
    if row['favorite_event'] == 1 and row['applyForPosition_event'] == 0:
        return 3
    if row['applyForPosition_event'] == 1:
        return 5
    return 0

def get_rating_matrix(job_data, interactions, threshold_users=5, threshold_items=5):


    #interactions['uui'] = interactions.userId.map(user_to_id)
    #interactions['iui'] = interactions.itemId.map(item_to_id)

    interactions_with_jobs = job_data.merge(interactions, on='itemId', how='left', indicator=True)
    rating_matrix = interactions_with_jobs[['userId', 'itemId', 'click', 'favorite_event', 'applyForPosition_event', 'industry', 'county', 'timestamp', 'applicationdeadline', 'published', 'state']]
    rating_matrix = rating_matrix[rating_matrix.click.eq(1) | rating_matrix.favorite_event.eq(1) | rating_matrix.applyForPosition_event.eq(1) ]
    rating_matrix['rating'] = rating_matrix.apply (lambda row: give_rating_based_on_interaction(row), axis=1)
    #rating_matrix = rating_matrix.dropna()[~(rating_matrix.rating == 0)]
    rating_matrix = rating_matrix[['userId', 'itemId', 'rating', 'industry', 'county', 'timestamp', 'applicationdeadline', 'published', 'state']]
    rating_matrix = rating_matrix.sort_values(by='rating')
    #rating_matrix = rating_matrix.drop_duplicates(['userId', 'itemId'], keep='last')

    #rating_matrix = rating_matrix.rename(columns={'uui': 'userId'})
    #rating_matrix = rating_matrix.rename(columns={'iui': 'itemId'})

    #rating_matrix.uidx = rating_matrix.uidx.astype(int)
    #rating_matrix.iidx = rating_matrix.iidx.astype(int)


    return rating_matrix

def map_ids(df, field, name):

    field_to_id = dict((j, i) for i,j in enumerate(df[field].unique()))
    id_to_field = dict((i, j) for i,j in enumerate(df[field].unique()))

    df[name] = df[field].map(field_to_id)
    return df, field_to_id

def get_item_features(rating_matrix):

    #job_interaction_matrix = job_interaction_matrix.sort_values(by='interactions', ascending=False)
    #job_interaction_matrix = job_interaction_matrix.reset_index()
    interaction_value_count = rating_matrix.iidx.value_counts()

    interaction_sum = interaction_value_count.sum() * 0.8
    interact_cumsum = 0
    short_head_item = 0

    for item, count in interaction_value_count.items():
        interact_cumsum += count
        if interact_cumsum >= interaction_sum:
            short_head_item = item
            break


    print(short_head_item, interaction_sum, interact_cumsum, interaction_value_count.sum())


    item_feature_df = pd.DataFrame(interaction_value_count).reset_index().rename(columns={'index': 'iidx', 'iidx': 'interactions'})
    short_head_idx = item_feature_df.index[item_feature_df['iidx'] == short_head_item ]
    print(short_head_idx[0])

    item_feature_df = item_feature_df.assign(feature='short_head')
    item_feature_df.loc[item_feature_df.index > short_head_idx[0], 'feature'] = 'long_tail'
    item_feature_df = item_feature_df[['iidx', 'feature', 'interactions']]
    item_feature_df['value'] = 1
    print(item_feature_df[item_feature_df.feature == 'short_head'].shape[0] / item_feature_df.shape[0], item_feature_df[item_feature_df.feature == 'long_tail'].shape[0] / item_feature_df.shape[0])


    item_feature_df = item_feature_df[item_feature_df.iidx.isin(rating_matrix.iidx.unique())]
    return item_feature_df, short_head_item

#Taken from: https://github.com/beta-team/beta-recsys/blob/master/beta_rec/datasets/data_split.py
def filter_user_item(df, min_u_c=5, min_i_c=5):
    """Filter data by the minimum purchase number of items and users.
    Args:
        df (DataFrame): interaction DataFrame to be processed.
        min_u_c (int): filter the items that were purchased by less than min_u_c users.
            (default: :obj:`5`)
        min_i_c (int): filter the users that have purchased by less than min_i_c items.
            (default: :obj:`5`)
    Returns:
        DataFrame: The filtered interactions
    """
    print(f"filter_user_item under condition min_u_c={min_u_c}, min_i_c={min_i_c}")
    print("-" * 80)
    print("Dataset statistics before filter")
    print(
        tabulate(
            df.agg(["count", "nunique"]),
            headers=df.columns,
            tablefmt="psql",
            disable_numparse=True,
        )
    )
    n_interact = len(df.index)

    while True:
        # Filter out users that have less than min_i_c interactions (items)
        if min_i_c > 0:
            df = filter_by_count(df, 'userId', 'itemId', min_i_c)

        # Filter out items that have less than min_u_c users
        if min_u_c > 0:
            df = filter_by_count(df, 'itemId', 'userId', min_u_c)

        new_n_interact = len(df.index)
        if n_interact != new_n_interact:
            n_interact = new_n_interact
        else:
            break  # no change
    check_data_available(df)
    print("Dataset statistics after filter")
    print(
        tabulate(
            df.agg(["count", "nunique"]),
            headers=df.columns,
            tablefmt="psql",
            disable_numparse=True,
        )
    )
    print("-" * 80)
    return df

def filter_by_count(df, group_col, filter_col, num):
    """Filter out the group_col column values that have a less than num count of filter_col.
    Args:
        df (DataFrame): interaction DataFrame to be processed.
        group_col (string): column name to be filtered.
        filter_col (string): column with the filter condition.
        num (int): minimum count condition that should be filter out.
    Returns:
        DataFrame: The filtered interactions.
    """
    ordercount = (
        df.groupby([group_col])[filter_col].nunique().rename("count").reset_index()
    )
    filter_df = df[
        df[group_col].isin(ordercount[ordercount["count"] >= num][group_col])
    ]
    return filter_df

def check_data_available(data):
    """Check if a dataset is available after filtering.
    Check whether a given dataset is available for later use.
    Args:
        data (DataFrame): interaction DataFrame to be processed.
    Raises:
        RuntimeError: An error occurred it there is no interaction.
    """
    if len(data.index) < 1:
        raise RuntimeError(
            "This dataset contains no interaction after filtering. Please check the default filter setup of this split!")