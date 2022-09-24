import datetime

import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.stats import expon
import pandas as pd
from tqdm import tqdm

pd.option_context('display.max_rows', 100,
                  'display.max_columns', None,
                  'display.precision', 3,
                  )

raw_probabilities = [0.013, 0.03123, 0.105, 0.2, 0.27, 0.26, 0.12]
days = [1, 2, 3, 4, 5, 6, 7]


def tweak_probabilities(probabilities):
    def twk_p(p):
        variation = random.uniform(0.3, 0.5)
        new_p = round(random.uniform(p * (1 - variation), p * (1 + variation)), 5)
        return new_p

    # probabilities = [twk_p(x) for x in probabilities]
    probabilities = [x * random.uniform(0.7, 1.3) for x in probabilities]
    return probabilities


def get_frequencies(sample_size, probabilities):
    frequencies = [int(round(x * sample_size, 0)) for x in probabilities]

    return frequencies


def get_week_day_list(days, frequencies):
    d_f_dict = dict(zip(days, frequencies))
    d_f_dict = [[k, v] for k, v in d_f_dict.items()]
    d_f_dict = [[x[0]] * x[1] for x in d_f_dict]
    d_f_dict = [j for i in d_f_dict for j in i]
    return d_f_dict


def get_transaction_list(sample_size, max_value_coef):
    transactions = expon.rvs(scale=max_value_coef, loc=1,
                             size=sample_size)  # scale reik empriškai išsibandyti, kad pasiekt reikiamą sumą
    transactions = [int(round(x, 0)) for x in transactions]
    return transactions


def equalize_list_length(col1, col2):
    if len(col1) > len(col2):
        col1 = col1[0:len(col2)]
    else:
        col2 = col2[0:len(col1)]
    return col1, col2


def get_day_n_transaction_df(days, raw_probabilities, max_value_coef):
    transaction_list = []
    while len(transaction_list) < 99000:
        weekly_sample_size = random.randint(50, 150)
        transaction_list = transaction_list + get_transaction_list(weekly_sample_size, max_value_coef)

    week_day_list = []
    while len(week_day_list) < 10000:
        weekly_sample_size = random.randint(50, 150)
        probabilities = tweak_probabilities(raw_probabilities)
        frequencies = get_frequencies(weekly_sample_size, probabilities)

        week_day_list = week_day_list + get_week_day_list(days, frequencies)


    week_day_list, transaction_list = equalize_list_length(week_day_list, transaction_list)
    df = pd.DataFrame({'week_day': week_day_list, 'transactions_eur': transaction_list})
    return df


def get_date_df():
    df_date = pd.DataFrame({'date': pd.date_range(start="2022-02-24", end="2022-09-21")})
    df_date['week_day'] = df_date['date'].dt.weekday + 1
    return df_date


fond_ltv = ['laisves_tv', 10000, 1000]
fond_BnY = ['Blue and Yellow', 5000, 500]
fond_1k = ['1K', 1000, 100]

df_all_fonds = pd.DataFrame(columns=['date', 'transactions_eur', 'fondas'])
for fond in [fond_1k, fond_ltv, fond_BnY]:
    fond_name = fond[0]
    sample_size = fond[1]
    max_value_coef = fond[2]

    df_days_n_trans = get_day_n_transaction_df(days=days,
                                               raw_probabilities=raw_probabilities,
                                               max_value_coef=max_value_coef)

    df_date = get_date_df()

    df_fond = df_date.merge(df_days_n_trans, on='week_day', how='right').drop('week_day', axis=1)
    df_fond['fondas'] = fond_name
    df_fond.to_csv(f'outputs/{fond_name}.tsv', sep='\t', index=False)

    df_all_fonds = pd.concat([df_fond, df_all_fonds])
    plt.hist(df_fond['date'], bins=100)
    plt.show()

df_all_fonds.to_csv(f'outputs/UKR_fonds.tsv', sep='\t', index=False)
