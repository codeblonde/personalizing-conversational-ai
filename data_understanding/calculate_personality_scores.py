"""
Here, methods to read, format and evaluate answers to the personality questionnaire are provided.
Scores per user and effects across the whole sample are calculated.
"""
import pandas as pd 
from visualization.visualizations import boxplot_trait_scores, scatterplot_interaction
from process_chats import get_summary, get_n_count, summarize_chats, sort_chat_messages, concat_and_save_message_strings


# basic processing of questionnaire

def read_personality_data(path):
    """
    Initial minor formatting is done here:
    A header with column names is added and the table pivoted.
    """
    # set column names
    columns = ["user_id", "question", "answer"]
    # load csv
    data_df = pd.read_csv(path, header=None, names=columns)
    # pivot table
    pivot_df = data_df.pivot_table(index="user_id", columns="question", values="answer")
    return pivot_df


def remove_fake_profiles(df_traits):
    """
    For testing the chat app, fake accounts were created and all questions answered with 1.
    These test accounts and other possible fake accounts are removed here.
    """
    # locate rows where all values are equal and create mask
    equals_mask = df_traits.eq(df_traits.iloc[:, 0], axis=0).all(axis=1)
    # invert mask
    inverted_mask = equals_mask != True
    # apply mask
    clean_df = df_traits[inverted_mask]
    return clean_df


def recode_answers(df_traits):
    """
    The BFI-S questionnaire contains positively and negatively poled questions.
    For evaluation, answers to negatively poled questions are re-coded.
    """
    poled_questions = [3, 6, 8, 15]
    for column in poled_questions:
        new_values = df_traits[column].replace([1, 2, 3, 4, 5, 6, 7], [7, 6, 5, 4, 3, 2, 1])
        df_traits[column].update(new_values)
    return df_traits


def calculate_scores_per_user(df_traits):
    """Calculate personality scores for each trait and user."""
    # dimensions and their respective questions (column indices)
    extra = [2, 6, 9]
    agree = [3, 7, 13]
    conscient = [1, 8, 12]
    openness = [4, 10, 14]
    neurotic = [5, 11, 15]
    # create empty data frame
    personality_df = pd.DataFrame()
    # add columns with mean score values
    personality_df['openness'] = df_traits[openness].mean(axis=1)
    personality_df['conscientiousness'] = df_traits[conscient].mean(axis=1)
    personality_df['extraversion'] = df_traits[extra].mean(axis=1)
    personality_df['agreeableness'] = df_traits[agree].mean(axis=1)
    personality_df['neuroticism'] = df_traits[neurotic].mean(axis=1)
    return personality_df


# collate questionnaire and chat data

def remove_superfluous_users(uniq_users, df_traits):
    """
    Remove users who filled out the personality questionnaire
    but did not participate in any chats from the personality data.
    """
    # compare user ids
    irregular_names = [n for n in df_traits['user_id'] if n not in uniq_users]
    print('users who did not participate in any of the chats:', irregular_names)
    # create boolean mask
    mask = df_traits['user_id'].isin(uniq_users)
    # apply mask and remove users
    clean_df = df_traits[mask]
    clean_df = clean_df.reset_index(drop=True)
    return clean_df


def map_extraversion_poles(df_traits):
    """
    Map scores to the traits' polar expressions.
    In the case of extraversion, scores equal to or above 3.5 are mapped to 'extrovert'.
    Scores below 3.5 are mapped to 'introvert'.
    """
    # select trait
    extraversion_scores = df_traits['extraversion']
    # create boolean mask based on half-way point of scale (3.5)
    mask = extraversion_scores >= 3.5
    # replace values
    expressions = mask.replace([True, False], ['extrovert', 'introvert'])
    # add to data frame
    df_traits['poles'] = expressions
    expression_df = df_traits.drop(['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism'],
                                   axis=1)
    # create dict from data frame
    expression_dict = pd.Series(expression_df.poles.values, index=expression_df.user_id).to_dict()
    print("Users and their respective extraversion poles:", expression_dict)
    return expression_dict


def get_interaction_message_lengths_scores(df_chats, df_traits, msg_lens):
    """
    Map extraversion scores to message lenths and assess possible factor interaction.
    """
    # map extraversion scores to users and create dict
    scores = df_traits.drop(['openness', 'conscientiousness', 'agreeableness', 'neuroticism'], axis=1)
    score_dict = pd.Series(scores.extraversion.values, index=scores.user_id).to_dict()
    # map scores and message lenghts to users in chat data frame
    df_chats['extraversion_scores'] = df_chats['user_id'].map(score_dict)
    interaction_df = pd.concat([df_chats['user_id'], msg_lens, df_chats['extraversion_scores']], axis=1)
    # for AVERAGE msg lengths grouped per user:
    # small_df = pd.concat([df_chats['user_id'], lens, df_chats['extraversion_scores']], axis=1)
    # grouped = small_df.groupby('user_id')
    # interaction = grouped.mean()
    return interaction_df


if __name__ == '__main__':
    # paths
    personality_path_in = '../ttas-data/ttas-user-answers.csv'
    personality_path_out = '../outputs/filtered-personality-scores.csv'
    chat_path_in = '../outputs/ttas-clean-chats.csv'
    chat_path_out = '../outputs/ttas-annotated-chats.csv'
    # read
    trait_data = read_personality_data(personality_path_in)
    chat_data = pd.read_csv(chat_path_in, sep=';') # header=None, names=columns)
    print(chat_data)
    # remove test profiles
    clean_answers = remove_fake_profiles(trait_data)
    # recode answers for calculation
    recoded_answers = recode_answers(clean_answers)
    # calculate scores
    trait_scores = calculate_scores_per_user(recoded_answers)
    trait_scores.reset_index(inplace=True)
    # compare with cleaned chat data and remove superfluous profiles
    unique_users, unique_chats, n_users, n_chats = get_n_count(chat_data)
    print(unique_users)
    filtered_scores = remove_superfluous_users(unique_users, trait_scores)
    # get personality score summary
    mean_scores = get_summary(filtered_scores.drop('user_id', axis=1))

    # map extraversion scores to pole expression labels
    extraversion_dict = map_extraversion_poles(filtered_scores)
    # annotate the cleaned chat data with the personality poles
    chat_data['extraversion_pole'] = chat_data['user_id'].map(extraversion_dict)
    # sort chats according to timestamp
    sorted_chats = sort_chat_messages(chat_data)
    # prepare for LIWC
    # prepare for LIWC
    concat_and_save_message_strings(sorted_chats)

    # save results
    sorted_chats.to_csv(chat_path_out, index=False, sep=';')
    filtered_scores.to_csv(personality_path_out, index=False, sep=';')

    # interaction between message lengths and extraversion trait scores
    message_lens, summary_chats, summary_messages = summarize_chats(sorted_chats)
    interaction = get_interaction_message_lengths_scores(sorted_chats, filtered_scores, message_lens)

    # visualize results
    boxplot_trait_scores(filtered_scores)
    scatterplot_interaction(interaction)

    print("""
    *******************************************************

                *** Manual step required ***

    To calculate Linguistic Style Matching scores, please
    refer to the Linguistic Inquiry and Word Count (LIWC) tool.
    Files for all individual chats have been created at 
    outputs/understanding/chats/ and are ready to be be analyzed 
    using the tool. The respective software can be found at:
    
                http://liwc.wpengine.com/
                
    Fees may apply.
    
    After analyzing the individual chats with the tool, save 
    results and continue with calculating the overall scores. 
    To do so, execute file:
    
                    calculate_liwc_results.py

    *******************************************************
    """)

