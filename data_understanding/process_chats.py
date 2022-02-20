"""Here, methods to filter out futile data points and create a statistical overview
 of the chat data set are provided."""

import re
import pandas as pd

from visualization.visualizations import histplot_messages


def read_chat_data(path):
    """Read in the data and add respective column names."""
    columns = ["chat_id", "user_id", "message", "timestamp"]
    chat_df = pd.read_csv(path, header=None, names=columns)
    print(chat_df.head())
    return chat_df


def filter_chats(df):
    """Filter data to only contain chats with appropriate number of turns."""
    # set threshold
    threshold = 3
    # group messages
    grouped = df.groupby('chat_id')
    # filter for message count
    filtered = grouped.filter(lambda x: x['message'].count() > threshold)
    return filtered


def get_summary(df):
    """Calculate the overall average, min and max values across sample."""
    # mean_scores= df.mean(axis=0)
    summary = df.agg(['min', 'mean', 'max'], axis=0)
    return summary


def summarize_chats(df):
    """Get statistical summaries of messages per chat and message lengths."""
    # group
    grouped = df.groupby('chat_id')
    # n messages per chat
    n_messages_per_chat = grouped['message'].count()
    # get min, max and average 
    summary_chats = get_summary(n_messages_per_chat)
    print('\nINFO: statistic summary of messages per chat:\n', summary_chats)
    # length of individual messages
    messages = df['message']
    # split individual messages and count words
    msg_lens = messages.str.split().str.len()
    # get min, max and average
    summary_msgs = get_summary(msg_lens)
    print('\nINFO: statistic summary of message lengths:\n', summary_msgs)
    return msg_lens, summary_chats, summary_msgs


def get_n_count(df):
    """
    Get count of unique users and chats.
    Note: There was one test case left in the chat data.
    """
    # get unique ids
    uniq_users = df.user_id.unique()
    uniq_chats = df.chat_id.unique()
    # get n unique ids
    n_users = df.user_id.nunique()
    n_chats = df.chat_id.nunique()
    # n_chats = len(unique_chats)
    print("""INFO: number of unique users: %i 
      number of unique chats: %i \n""" % (n_users, n_chats))
    return uniq_users, uniq_chats, n_users, n_chats


def clean_messages(df):
    """ Clean dataframe from emoticons and other special tokens"""
    emoticons = re.compile('(\:\w+\:|\<[\/\\]?3|[\(\)\\\D|\*\$][\-\^]?[\:\;\=]|[\:\;\=B8][\-\^]?[3DOPp\@\$\*\\\)\(\/\|])(?=\s|[\!\.\?]|$)')
    special_chars = re.compile('[$&+:;=|"@#<>^*()%/_-]')
    # apply regex to df
    df['message'] = df['message'].apply(lambda x: emoticons.sub(r'.', x))
    df['message'] = df['message'].apply(lambda x: special_chars.sub(r'', x))
    return df


def sort_chat_messages(df):
    """Sort messages by chat id and timestamp"""
    # convert string to datetime object
    df['timestamp'] = pd.to_datetime(df['timestamp'], format="%Y-%m-%d %H:%M:%S.%f")
    # sort and group by chat id first
    df = df.sort_values(['chat_id'], ascending=True)
    df = df.groupby(['chat_id'], sort=False)
    # then sort by datetime object
    df = df.apply(lambda x: x.sort_values(['timestamp'], ascending=True))
    df = df.reset_index(drop=True)
    return df


def concat_and_save_message_strings(df, path='../outputs/chats/'):
    # group, sort and concat message strings
    for group, frame in df.groupby('chat_id'):
        frame = frame.sort_values(['user_id'])
        # concat messages per user
        strings = frame.groupby(['user_id'])['message'].apply(' '.join).reset_index()
        strings.to_csv(path+'{}.csv'.format(group), sep=';', index=False)


if __name__ == '__main__':
    chats_input_path = '../ttas-data/ttas-complete-chats.csv'
    chats_output_path = '../outputs/ttas-clean-chats.csv'
    # read data
    chat_data = read_chat_data(chats_input_path)
    print('INFO: Results pre filtering:')
    unique_users, unique_chats, number_users, number_chats = get_n_count(chat_data)
    # filter data
    filtered_chats = filter_chats(chat_data)
    clean_chats = clean_messages(filtered_chats)
    # get n unique users and chats
    print('INFO: Results post filtering:')
    unique_users_clean, unique_chats_clean, number_users_clean, number_chats_clean = get_n_count(clean_chats)
    # get n messages per chats, message lengths
    message_lens, chat_summary, summary_messages = summarize_chats(clean_chats)
    # save
    clean_chats.to_csv(chats_output_path, index=False, sep=';')
    # visualize
    histplot_messages(message_lens)
    print("""
    *******************************************************
    
                *** Manual cleaning advised ***
                
    Please manually correct spelling, casing, abbreviations,
    foreign language use and superfluous white spaces
    to enhance quality of the data.
    
    The corresponding file can be found under:
    
             outputs/understanding/ttas-clean-chats.csv
             
    After cleaning continue with calculating the respective
    personality scores. To do so, execute file:
     
                calculate_personality_score.py
        
    *******************************************************
    """)
