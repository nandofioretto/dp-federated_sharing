from __init__ import dataset_path
import pandas as pd
import numpy as np
from utils import scale
from sklearn import preprocessing

def load_dataset(name, pfeat=None, to_numeric=True, classes=[0, 1]):
    if name == 'census':
        dataset = pd.read_csv(dataset_path + 'census/adult_income_dataset.csv', na_values='?', skipinitialspace=True)
        x_feat = ['age', 'workclass', 'education-num', 'marital-status',
                  'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
                  'hours-per-week', 'native-country']
        y_feat = 'fnlwgt'
        del dataset['education']
        cat = ['workclass', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']

        p_feat = 'education-num' if pfeat is None else pfeat
    elif name == 'credit':
        dataset = pd.read_csv(dataset_path + 'to_organize/default_payment_of_credit_card_clients.csv', na_values='?', skipinitialspace=True)
        x_feat = ['LIMIT_BAL','SEX','EDUCATION','MARRIAGE','AGE','PAY_0','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6',
                  'BILL_AMT1','BILL_AMT2','BILL_AMT3','BILL_AMT4','BILL_AMT5','BILL_AMT6','PAY_AMT1','PAY_AMT2',
                  'PAY_AMT3', 'PAY_AMT4','PAY_AMT5','PAY_AMT6']
        y_feat = 'payment'
        p_feat = 'LIMIT_BAL' if pfeat is None else pfeat
        dataset.drop('ID', 1, inplace=True)

        cat = ['SEX','EDUCATION','MARRIAGE','PAY_0','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6']
    elif name == 'bike_day_h':
        dataset = pd.read_csv(dataset_path + 'bike_sharing/hour.csv',skipinitialspace=True)
        for feat in ['instant','dteday','yr','casual','registered']:
            dataset.drop(feat, 1, inplace=True)
        x_feat = ['season', 'mnth', 'hr', 'holiday', 'weekday', 'workingday', 'weathersit', 'temp', 'atemp',
                  'hum', 'windspeed']
        y_feat = 'cnt'
        p_feat = 'temp' if pfeat is None else pfeat
        #dataset.loc[:, y_feat] = normalize(dataset[y_feat].values, classes[0], classes[1])

        cat = ['season','mnth','holiday','weekday','workingday','weathersit']
    elif name == 'bike_day_d':
        dataset = pd.read_csv(dataset_path + 'bike_sharing/day.csv', skipinitialspace=True)
        for feat in ['instant', 'dteday', 'yr', 'casual', 'registered']:
            dataset.drop(feat, 1, inplace=True)
        x_feat = ['season', 'mnth', 'holiday', 'weekday', 'workingday', 'weathersit', 'temp', 'atemp', 'hum',
                  'windspeed']
        y_feat = 'cnt'
        p_feat = 'temp' if pfeat is None else pfeat
        #dataset.loc[:, y_feat] = normalize(dataset[y_feat].values, classes[0], classes[1])

        cat = ['season','mnth','holiday','weekday','workingday','weathersit']

    elif name == 'diabetes':
        dataset = pd.read_csv(dataset_path + 'medical/pima_indian_diabetes_dataset.csv', skipinitialspace=True)
        x_feat = ['num_preg','glucose_conc','diastolic_bp','thickness','insulin','bmi','diab_pred','age','skin']
        y_feat = 'diabetes'
        p_feat = 'glucose_conc' if pfeat is None else pfeat
        dataset.loc[:, 'diabetes'] = dataset['diabetes'].values.astype(int)

        cat = []
    elif name == 'cancer':
        dataset = pd.read_csv(dataset_path + 'medical/wisconsin_diagnostic_breast_cancer.csv', skipinitialspace=True)
        x_feat = ['diagnosis','radius_mean','texture_mean','perimeter_mean','area_mean','smoothness_mean',
                  'compactness_mean','concavity_mean','concave_points_mean','symmetry_mean','fractal_dimension_mean',
                  'radius_se','texture_se','perimeter_se','area_se','smoothness_se','compactness_se','concavity_se',
                  'concave_points_se','symmetry_se','fractal_dimension_se','radius_worst','texture_worst',
                  'perimeter_worst','area_worst','smoothness_worst','compactness_worst','concavity_worst',
                  'concave_points_worst','symmetry_worst','fractal_dimension_worst']
        y_feat = 'diagnosis'
        p_feat = 'radius_mean' if pfeat is None else pfeat
        dataset.drop('id', 1, inplace=True)
        cat = []
    elif name == 'news':
        dataset =  pd.read_csv(dataset_path + 'OnlineNewsPopularity/OnlineNewsPopularity.csv', skipinitialspace=True)
        for feat in ['timedelta', 'url','n_non_stop_words','weekday_is_monday', 'weekday_is_tuesday',
                     'weekday_is_wednesday', 'weekday_is_thursday','weekday_is_friday', 'weekday_is_saturday',
                     'weekday_is_sunday']:
            dataset.drop(feat, 1, inplace=True)

        x_feat = ['n_tokens_title', 'n_tokens_content', 'n_unique_tokens',
                  'n_non_stop_unique_tokens', 'num_hrefs', 'num_self_hrefs', 'num_imgs', 'num_videos',
                  'average_token_length', 'num_keywords', 'data_channel_is_lifestyle', 'data_channel_is_entertainment',
                  'data_channel_is_bus', 'data_channel_is_socmed', 'data_channel_is_tech', 'data_channel_is_world',
                  'kw_min_min', 'kw_max_min', 'kw_avg_min', 'kw_min_max', 'kw_max_max', 'kw_avg_max', 'kw_min_avg',
                  'kw_max_avg', 'kw_avg_avg', 'self_reference_min_shares', 'self_reference_max_shares',
                  'self_reference_avg_sharess', 'is_weekend', 'LDA_00', 'LDA_01', 'LDA_02', 'LDA_03', 'LDA_04', 'global_subjectivity',
                  'global_sentiment_polarity', 'global_rate_positive_words', 'global_rate_negative_words',
                  'rate_positive_words', 'rate_negative_words', 'avg_positive_polarity', 'min_positive_polarity',
                  'max_positive_polarity', 'avg_negative_polarity', 'min_negative_polarity', 'max_negative_polarity',
                  'title_subjectivity', 'title_sentiment_polarity', 'abs_title_subjectivity',
                  'abs_title_sentiment_polarity']
        y_feat = 'shares'
        p_feat = 'n_tokens_content' if pfeat is None else pfeat

        cat = ['data_channel_is_lifestyle','data_channel_is_entertainment','data_channel_is_bus',
               'data_channel_is_socmed','data_channel_is_tech','data_channel_is_world',
               'is_weekend']

    dataset = dataset.dropna().reset_index(drop=True)
    ########################
    ## Data Preprocessing and Agent data allocator
    ########################
    if to_numeric:
        lb_make = preprocessing.LabelEncoder()
        obj_df = dataset.select_dtypes(include=['object']).copy()
        for feat in list(obj_df.columns):
            dataset.loc[:, feat] = lb_make.fit_transform(obj_df[feat])

    dataset.loc[:, y_feat] = scale(dataset[y_feat].values, classes[0], classes[1]).astype(type(classes[0]))

    # move target values (y) in the first column
    targetcol = dataset[y_feat]
    dataset.drop(labels=[y_feat], axis=1, inplace=True)
    dataset.insert(0, y_feat, targetcol)

    return dataset, x_feat, y_feat, p_feat, cat


''' Partitions a dataset into N datasets to feed to different agents'''
def allocate_data_to_agents(data, n_agts, p_feat, type='lin'):
    from utils import to_probability
    '''
        :param type: lin - linear space between attributes or
                     rnd - randomly distributed
    '''
    attr = p_feat
    if type == 'lin':
        means = np.linspace(np.min(data[[attr]].values),
                            np.max(data[[attr]].values),
                            n_agts + 2)[1:-1]
    elif type == 'rnd':
        means = np.random.uniform(np.min(data[[attr]].values),
                          np.max(data[[attr]].values),
                          n_agts)

    agt_data_rows = {i: [] for i in range(n_agts)}
    for index, row in data.iterrows():
        Pr = to_probability(1 / (abs(row[attr] - means)+0.000001))
        i = np.random.choice(range(n_agts), p=Pr)
        agt_data_rows[i].append(index)

    # for i in range(n_agts):
    #     if data.iloc[agt_data_rows[i], :]

    return {i: data.iloc[agt_data_rows[i], :].reset_index(drop=True) for i in range(n_agts)}
