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


def get_week_day_list(days, weekly_sample_size, raw_probabilities):
    def tweak_probabilities(probabilities):
        def twk_p(p):
            variation = random.uniform(0.3, 0.5)
            new_p = round(random.uniform(p * (1 - variation), p * (1 + variation)), 5)
            return new_p

        # probabilities = [twk_p(x) for x in probabilities]
        probabilities = [x * random.uniform(0.7, 1.3) for x in probabilities]
        return probabilities

    def get_frequencies(weekly_sample_size, probabilities):
        frequencies = [int(round(x * weekly_sample_size, 0)) for x in probabilities]

        return frequencies

    def multiply_weekdays_by_frequeces(days, frequencies):
        d_f_dict = dict(zip(days, frequencies))
        d_f_dict = [[k, v] for k, v in d_f_dict.items()]
        d_f_dict = [[x[0]] * x[1] for x in d_f_dict]
        d_f_dict = [j for i in d_f_dict for j in i]
        return (d_f_dict)

    probabilities = tweak_probabilities(raw_probabilities)
    frequencies = get_frequencies(weekly_sample_size, probabilities)
    d_f_dict = multiply_weekdays_by_frequeces(days, frequencies)
    return d_f_dict


def get_transaction_list(sample_size, max_eur_coef):
    transactions = expon.rvs(scale=max_eur_coef, loc=1,
                             size=sample_size)  # scale reik empriškai išsibandyti, kad pasiekt reikiamą sumą
    transactions = [int(round(x, 0)) for x in transactions]
    return transactions


def equalize_list_length(col1, col2):
    if len(col1) > len(col2):
        col1 = col1[0:len(col2)]
    else:
        col2 = col2[0:len(col1)]
    return col1, col2


def get_day_n_transaction_df(days, raw_probabilities, max_eur_coef, week_count, total_sample_size):
    def get_weekly_sample_size(total_sample_size, week_count):
        avg_weekly_sample_size = total_sample_size / week_count
        min_rand_smp_size = round(avg_weekly_sample_size * 0.5, 0)
        max_rand_smp_size = round(avg_weekly_sample_size * 1.5, 0)

        weekly_sample_size = random.randint(min_rand_smp_size, max_rand_smp_size)
        return weekly_sample_size

    weekly_sample_size = get_weekly_sample_size(total_sample_size, week_count)

    transaction_list = get_transaction_list(weekly_sample_size, max_eur_coef)
    week_day_list = get_week_day_list(days, weekly_sample_size, raw_probabilities)

    transaction_list, week_day_list = equalize_list_length(transaction_list, week_day_list)

    df = pd.DataFrame({'week_day': week_day_list, 'transactions_eur': transaction_list})
    return df


def get_date_df():
    df_date = pd.DataFrame({'date': pd.date_range(start="2022-02-24", end="2022-09-21")})
    df_date['week'] = df_date['date'].dt.isocalendar().week
    df_date['week_day'] = df_date['date'].dt.weekday + 1
    return df_date


class fond_details:
    def __init__(self, name, total_sample_size, max_eur_coef):
        self.name = name
        self.total_sample_size = total_sample_size
        self.max_eur_coef = max_eur_coef


raw_probabilities = [0.013, 0.03123, 0.105, 0.2, 0.27, 0.26, 0.12]
days = [1, 2, 3, 4, 5, 6, 7]

fond_ltv = fond_details('laisves_tv', 10000, 1000)
fond_BnY = fond_details('Blue and Yellow', 5000, 500)
fond_1k = fond_details('1K', 1000, 100)

df_all_fonds = pd.DataFrame(columns=['date', 'transactions_eur', 'fondas'])
for fond in [fond_1k, fond_ltv, fond_BnY]:
    fond_name = fond.name
    fund_total_sample_size = fond.total_sample_size
    max_eur_coef = fond.max_eur_coef

    weeks = get_date_df()['week'].tolist()  # Defining list of iso_year_weeks in date range
    df_week = get_date_df()  # Df of Weekly dates - will take weekly_date data from here
    df_fond = get_date_df().iloc[0:0]  # an empty df to attach the joint data of dates and transactions

    for week in tqdm(weeks):
        # Making weekly datasets of dates and transactions
        df_fond_week = df_week.loc[df_week['week'] == week, :]
        df_days_n_trans = get_day_n_transaction_df(days=days,
                                                   raw_probabilities=raw_probabilities,
                                                   max_eur_coef=max_eur_coef,
                                                   week_count=len(weeks), total_sample_size=fund_total_sample_size)
        # Joining them
        df_fond_week = df_fond_week.merge(df_days_n_trans, on='week_day', how='right').drop('week_day', axis=1)
        # Attaching weekly df to the main df of single Fond
        df_fond = pd.concat([df_fond, df_fond_week])
    # Cleaning, supplementation, writing
    df_fond['fondas'] = fond_name
    df_fond = df_fond.loc[df_fond['date'].notna()]
    df_fond.to_csv(f'outputs/{fond_name}.tsv', sep='\t', index=False)

    # Attaching to the main df od ALL Funds
    df_all_fonds = pd.concat([df_fond, df_all_fonds])

    # Plotting
    plt.hist(df_fond['date'], bins=len(weeks))
    plt.savefig(f"outputs/graphs/{fond_name}_transacton_count-per_week.png")
    plt.show()

    plt.hist(df_fond['transactions_eur'], bins=100)
    plt.savefig(f"outputs/graphs/{fond_name}_transacton_count-by_size.png")
    plt.show()

# Writing
df_all_fonds.to_csv(f'outputs/UKR_fonds.tsv', sep='\t', index=False)
