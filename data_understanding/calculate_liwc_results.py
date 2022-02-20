"""Here, methods to assess Linguistic Style Matching in the chat dialogues are provided."""

import glob
import os
import pandas as pd
import numpy as np
from scipy import stats
from visualization.visualizations import boxplot_lsm_scores


def calculate_lsm(path):
    """Calculate scores for Linguistic Style Matching for each chat.
    Calculations are based on Gonzales et al. 2010
    e.g. ppLSM = 1(|pp1 - pp2|/(pp1 + pp2))"""
    # columns of interest for calculation
    cols = ['auxverb', 'article', 'adverb', 'ppron', 'ipron', 'prep', 'negate', 'conj', 'posemo', 'negemo']
    scores_df = pd.DataFrame()
    # iterate over files
    for file_path in glob.iglob(path + '*.csv'):
        chat_id = os.path.basename(file_path)[:-4]
        df = pd.read_csv(file_path, sep=";")
        df = df[cols]
        # convert strings to floats
        # Note: issue with converting non-float numbers (e.g. WC 45 -> NaN)
        df = df.apply(lambda x: x.str.replace(',', '.').astype(float), axis=1)
        diff_scores = (df.loc[0, :] - df.loc[1, :]).abs()
        sum_scores = df.sum(axis=0)
        lsm_score = 1.0 - (diff_scores / sum_scores)
        # fill NaNs with score 1: NaNs result from 0 / 0 division
        # since the scores are basically a match, they are assigned a score of 1
        # Note: this decision did not make much of difference in the end results
        lsm_score = lsm_score.fillna(1)
        lsm_score['average'] = np.mean(lsm_score[:-2])
        lsm_score['chat_id'] = chat_id
        scores_df = scores_df.append(lsm_score, ignore_index=True)
    # re-order
    names = scores_df.pop('chat_id')
    avgs = scores_df.pop('average')
    scores_df.insert(0, 'chat_id', names)
    scores_df.insert(1, 'overall', avgs)
    score_summary = scores_df.describe()
    return scores_df, score_summary


def trait_by_chat(path):
    """Match chat ids to a personality trait.
    Introvert means at least one introverted speaker participated in the chat.
    Note: chats were either mixed or had between only extroverts."""
    df = pd.read_csv(path, sep=";")
    chats = pd.unique(df['chat_id'])
    introvert_chats = [group for group, df in df[df['extraversion_pole'].str.contains('introvert')].groupby('chat_id')]
    personality_dict = {}
    for chat in chats:
        if chat in introvert_chats:
            personality_dict[chat] = 'introvert'
        else:
            personality_dict[chat] = 'extrovert'
    return personality_dict


def match_trait_to_scores(df_lsm, dict_traits):
    """Extract traits and chat ids"""
    df_lsm['trait'] = df_lsm['chat_id'].map(dict_traits)
    score_by_trait = df_lsm.groupby('trait').mean()
    # Note: no (sig) difference between groups, BUT may differ with actual trait scores
    # to average the scores of both speakers:
    # mixed_chats = score_by_trait.loc['introvert', :].tolist()
    # extro_chats = score_by_trait.loc['extrovert', :].tolist()
    return score_by_trait


def compare_scores(df):
    """Compare lsm scores for chats with and without introverts present"""
    df = df.drop('overall', axis=1)
    mixed_chats = df.loc['introvert', :].values.tolist()
    extro_chats = df.loc['extrovert', :].values.tolist()
    t_sig = stats.ttest_ind(extro_chats, mixed_chats)
    print('\nINFO: Results t-test for chats between only extroverts compared to mixed personality group chats:\n', t_sig)
    return t_sig


if __name__ == '__main__':
    scores, overview = calculate_lsm('../outputs/understanding/liwc/')

    trait_dict = trait_by_chat('../outputs/ttas-annotated-chats.csv')
    matched = match_trait_to_scores(scores, trait_dict)
    print('\nINFO: Results for LSM scores overall and for each factor grouped by personality trait:\n', matched)
    ttest = compare_scores(matched)

    #save
    scores.to_csv('../outputs/liwc-scores.csv', sep=';', index=False)
    overview.to_csv('../outputs/liwc-overall-stats.csv', sep=';')

    #visualize
    boxplot_lsm_scores(scores)

    print("""
    *******************************************************

                    *** Congratulations ***

    Basic cleaning and analysis of the data are done. 
    You should have a good understanding of the data now.
    Please continue with preparing the cleaned chats for
    the modeling pipeline. To do so, please switch to the
    modeling directory and execute file:
     
                    add_context_columns.py

    *******************************************************
    """)