"""Here, methods to prepare the dataset for training are provided"""
import pandas as pd
from itertools import chain
from collections import defaultdict

from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import (AutoModelWithLMHead, AutoTokenizer)


# define special tokens
SPECIAL_TOKENS = ['<bos>', '<eos>', '<speaker1>', '<speaker2>', '<introvert>', '<extrovert>', '<pad>']
ATTR_TO_SPECIAL_TOKEN = {'bos_token': '<bos>', 'eos_token': '<eos>', 'pad_token': '<pad>',
                         'additional_special_tokens': ['<speaker1>', '<speaker2>', '<introvert>', '<extrovert>']}
MODEL_INPUTS = ['input_ids', 'mc_token_ids', 'lm_labels', 'mc_labels', 'token_type_ids']
PADDED_INPUTS = ['input_ids', 'lm_labels', 'token_type_ids']


# TODO: check Trainer for modeling


def tokenize_dataset(df, tokenizer):
    """Tokenize string values specified columns
    Note: dbmbz pre-trained tokenizer cannot be applied to batches of sentences
    tokenize: separates string into list of words and punctuation marks
    convert_tokens_to_ids: convert words into indices of vocabulary entries"""
    print('INFO: Tokenizing messages ...')
    # tokenize and encode
    cols = ['message', 'distractor_1', 'distractor_2', 'context_0', 'context_1', 'context_2']
    for name in cols:
        df[name] = df[name].apply(tokenizer.tokenize)
        df[name] = df[name].apply(tokenizer.convert_tokens_to_ids)
    return df


def split_dataframe(df):
    """Concatenate candidates and contexts after tokenization
    -> last response is ground truth
    Note: token id 255 is an empty string and should be removed
    Split into train and test set
    test_size is set to 0.15 since the dataset is quite small"""
    print('INFO: Splitting dataset ...')
    new_df = pd.DataFrame()
    new_df['trait'] = df['extraversion_pole']
    new_df['candidates'] = df.apply(lambda x: [x['distractor_1']] + [x['distractor_2']] + [x['message']], axis=1)
    new_df['context'] = df.apply(lambda x: [x['context_2']] + [x['context_1']] + [x['context_0']], axis=1)
    new_df['context'] = [[msg for msg in li if msg != [225]] for li in new_df['context']]
    # split in train and test
    train, test = train_test_split(new_df, test_size=0.15, random_state=0, stratify=new_df[['trait']])
    train.reset_index(drop=True, inplace=True)
    test.reset_index(drop=True, inplace=True)
    print('INFO: Train and test samples:', train.shape, test.shape)
    return train, test


def pad_dataset(dataset, padding=0):
    """Pad Dataset.
    Note: LM Labels are padded differently
    max length of history + response  = 443 tokens
    training size = 512 for dbmdz
    training size = 1024 for GerPT"""
    print('INFO: Padding inputs ...')
    #max_l = max(len(x) for x in dataset['input_ids'])
    max_l = 512
    for name in PADDED_INPUTS:
        dataset[name] = [x + [padding if name != 'lm_labels' else -100] * (max_l - len(x)) for x in dataset[name]]
    return dataset


def add_special_token(model, tokenizer):
    """Add special tokens to training and tokenizer.
    Check with pretrained tokens."""
    n_added_tokens = tokenizer.add_special_tokens(ATTR_TO_SPECIAL_TOKEN)
    if n_added_tokens > 0:
        model.resize_token_embeddings(new_num_tokens=len(tokenizer))


def build_inputs(tokenizer, trait, history, response, lm_labels=False, with_eos=True):
    """Build modeling sequences from pole, history and response segments
    - history = list of previous utterances as list of list of token ids / words
    - response = list of token ids / words for gold or distractor response
    - trait = trait special token
    Returns dict"""
    # convert special token symbols to token ids
    bos, eos, speaker1, speaker2, introvert, extrovert = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[:-1])
    # set trait poles to respective tokens / token ids
    if trait == 'introvert':
        pole = introvert
    elif trait == 'extrovert':
        pole = extrovert
    # create sequences
    sequence = [[bos] + [pole]] + history + [response + ([eos] if with_eos else [])]
    sequence = [sequence[0]] + [[speaker2 if (len(sequence)-i) % 2 else speaker1]
                                + s for i, s in enumerate(sequence[1:])]
    instance = dict()
    instance['input_ids'] = list(chain(*sequence))
    instance['token_type_ids'] = [speaker2 if i % 2 else speaker1 for i, s in enumerate(sequence) for _ in s]
    instance['mc_token_ids'] = len(instance['input_ids']) - 1
    instance['lm_labels'] = [-100] * len(instance['input_ids'])
    if lm_labels:
        instance['lm_labels'] = ([-100] * sum(len(s) for s in sequence[:-1])) + [-100] + sequence[-1][1:]
    return instance


def build_dataset(df, tokenizer, train_set=True, distributed=False):
    """
    Transforms the input dataframe or dict into a Tensor Dataset.
    Note: Distributed Training is only supported on Linux and Windows
          For support on Mac library needs to be compiled from source
    """
    print('INFO: Building dataset')
    dataset = defaultdict(list)
    n_candidates = 3
    max_history = 2
    if train_set:
        n_candidates = n_candidates
    else:
        n_candidates = 1
    # create instance for each candidate response
    print('INFO: Building sequences ...')
    for i, row in df.iterrows():
        trait = row['trait']
        history = row['context'][-(2*3+1):]
        candidates = row['candidates']
        for j, candidate in enumerate(candidates[-n_candidates:]): # possible error -> gold response has index 2 ?
            lm_labels = bool(j == n_candidates-1)
            instance = build_inputs(tokenizer, trait, history, candidate, lm_labels)
            for input_name, input_array in instance.items():
                dataset[input_name].append(input_array)
            dataset['mc_labels'].append(n_candidates - 1) # label == 2?
            dataset['n_candidates'] = n_candidates
    # pad
    padded_dataset = pad_dataset(dataset, padding=tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[-1]))

    # convert to tensors
    print('INFO: Converting input sequences into tensors ...')
    tensor_set = []
    for input_name in MODEL_INPUTS:
        tensor = torch.tensor(padded_dataset[input_name])
        tensor = tensor.view((-1, dataset['n_candidates']) + tensor.shape[1:])
        tensor_set.append(tensor)

    #build tensor data set
    batchsize = 4
    tensor_dataset = TensorDataset(*tensor_set) # TODO: resolve size mismatch error
    sampler = DistributedSampler(tensor_dataset) if distributed else None
    loader = DataLoader(tensor_dataset, sampler=sampler, batch_size=batchsize, shuffle=False)
    print('INFO: Dataset (Batch, Candidates, Seq Length):{}'.format(tensor_dataset.tensors[0].shape))
    return loader, sampler


def get_data_loaders(data_path, tokenizer, model):
    """ Load, tokenize and split data and build tensor datasets for training """
    data = pd.read_csv(data_path, sep=";")
    data = data.drop(['chat_id', 'user_id'], axis=1)
    add_special_token(model, tokenizer)
    tokenized_chats = tokenize_dataset(data, tokenizer)
    train, test = split_dataframe(tokenized_chats)
    train_loader, train_sampler = build_dataset(train, tokenizer)
    test_loader, test_sampler = build_dataset(test, tokenizer, train_set=False)
    return train_loader, train_sampler, test_loader, test_sampler



#if __name__ == '__main__':
    #data = '../outputs/context-chats.csv'
    #tokenizer = AutoTokenizer.from_pretrained('dbmdz/german-gpt2')
    #training = AutoModelWithLMHead.from_pretrained('dbmdz/german-gpt2')
    #train_loader, train_sampler, test_loader, test_sampler = get_data_loaders(data, tokenizer, training)