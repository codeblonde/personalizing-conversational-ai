"""Here, methods to process the data and create context and distractor data are provided."""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def change_userid_to_speakerid(df):
    """Change made up user ids to speaker 1 and speaker 2 respectively"""
    uniq_user_id = df.user_id.unique()
    chat = df['chat_id']
    if len(uniq_user_id) < 2:
        print('WARNING: Conversation with only 1 speaker detected. Please remove conversation:', chat)
    else:
    # map to dict
        speaker_dict = {uniq_user_id[0]: 'speaker1', uniq_user_id[1]: 'speaker2'}
        df['user_id'] = df['user_id'].map(speaker_dict)
    return df


def clean_up_turns(df):
    """
    Find incidents where the same speaker sent two succeeding messages and concatenate into one message/turn.
    """
    # create boolean mask
    # shift(-1) compares to row beneath
    # shift(1) compares to row above
    # mask = df['user_id'].shift() == df['user_id'] #and (df['chat_id'].shift(-1) == df['chat_id']))
    mask = ((df['user_id'].shift() == df['user_id']) & (df['chat_id'].shift() == df['chat_id']))
    # get indices
    mask_ind = mask.reset_index(drop=True)
    mask_ind = mask_ind[mask_ind].index.tolist()
    concat_ind = [ind - 1 for ind in mask_ind]
    ind_tuple = zip(concat_ind, mask_ind)
    # concatenate messages / turns
    for tpl in ind_tuple:
        df.iloc[tpl[0]]['message'] = df.iloc[tpl[0]]['message'] + ' ' + df.iloc[tpl[1]]['message']
    # drop redundant messages
    df_clean = df[~mask]
    return df_clean


def create_context_cols(df):
    """Create context columns in the data frame.
    Note: Distractors are picked randomly from predefined distractor_sents.
    For a larger dataset they should be picked randomly from the dataset itself"""
    distractor_sents = pd.Series(['Das tur mir leid.', 'Das hab ich nicht verstanden.', 'Super cool!', 'Wie meinst du das?',
                        'Ich liebe Eis.', 'Ich bin vegan.', 'Was ist dein Lieblingsessen?', 'Was ist dein Hobby?',
                        'Ich mag Suppe.', 'Was hast du morgen so vor?'])
    df['context_0'] = df['message'].shift(1, fill_value='Hi!')
    df['context_1'] = df['message'].shift(2, fill_value=' ')
    df['context_2'] = df['message'].shift(3, fill_value=' ')
    
    df['distractor_1'] = distractor_sents[np.random.randint(0, len(distractor_sents), len(df)).tolist()].tolist()
    df['distractor_2'] = distractor_sents[np.random.randint(0, len(distractor_sents), len(df)).tolist()].tolist()
    return df


def format_context_response_table(df):
    # concat messages
    df_turns = clean_up_turns(df)
    print(df_turns)
    # change usernames
    df_speaker = df_turns.groupby('chat_id').apply(change_userid_to_speakerid)
    # create context columns
    df_context = df_speaker.groupby('chat_id').apply(create_context_cols)
    return df_context


def table_to_nested_dict(df):
    """Create a nested dict of the data that can be saved to json file.
    The two main keys are the two extraversion trait poles, each key holds the individual messages,
    their respective chat history and distractor replies.
    Note: Not used in the final pipeline."""
    df['candidates'] = df.apply(lambda x: [x['distractor_1']] + [x['distractor_2']] + [x['message']], axis=1)
    df['context'] = df.apply(lambda x: [x['context_2']] + [x['context_1']] + [x['context_0']], axis=1)
    df['context'] = [[msg for msg in li if msg != ' '] for li in df['context']]

    keys = ['personality', 'utterances']
    data = {'train': [], 'test': []}
    grouped = df.groupby('extraversion_pole')
    for group, frame in grouped:
        train, test = train_test_split(frame, test_size=0.15)
        print(len(train), len(test))
        personality_dict = dict.fromkeys(keys)
        personality_dict['personality'] = group
        personality_dict['utterances'] = []
        for idx, row in train.iterrows():
            sub_dict = dict()
            sub_dict['candidates'] = row['candidates']
            sub_dict['history'] = row['context']
            personality_dict['utterances'].append(sub_dict)
        data['train'].append(personality_dict)
        for idx, row in test.iterrows():
            sub_dict = dict()
            sub_dict['candidates'] = row['candidates']
            sub_dict['history'] = row['context']
            personality_dict['utterances'].append(sub_dict)
        data['test'].append(personality_dict)

    return data


if __name__ == '__main__':
    # create context and distractor columns
    chats = pd.read_csv('../outputs/ttas-annotated-chats.csv', sep=";")
    contextual_df = format_context_response_table(chats)
    contextual_df = contextual_df.drop(['timestamp'], axis=1)
    contextual_df.reset_index(drop=True, inplace=True)
    contextual_df.to_csv('../outputs/training/context-chats.csv', sep=';', index=False)

    print("""
    *******************************************************

                    *** Dataframe complete ***

    Context and Distractor columns have been added to the
    data frame. Please continue with preprocessing the data  
    and training. To do so, execute file:

                        train.py

    *******************************************************
    """)


