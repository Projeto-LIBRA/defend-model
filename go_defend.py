# coding=utf-8
from __future__ import unicode_literals

import json
import os
from urllib.request import urlopen, Request
from pathlib import Path
from defend import Defend
import pandas as pd
from bs4 import BeautifulSoup
import re

import numpy as np
from sklearn.model_selection import train_test_split
from nltk import tokenize
from keras.utils.np_utils import to_categorical
import pickle

import nltk
nltk.download('punkt')

DATA_PATH = 'data'
SAVED_MODEL_DIR = 'saved_models'
SAVED_MODEL_FILENAME = 'polifact_Defend_model.h5'
EMBEDDINGS_PATH = 'archive'


def clean_str(string):
    """
    Tokenization/string cleaning for dataset
    Every dataset is lower cased except
    """
    string = re.sub(r"\\", "", string)
    string = re.sub(r"\'", "", string)
    string = re.sub(r"\"", "", string)
    return string.strip().lower()

def get_dataset(file_name):
    labels_folder = ['fake', 'real']
    comments = []
    contents = []
    labels = []
    texts = []
    ids = []
    tmp_comments = []
    for label in labels_folder:
        path = './dataset/fakenews_dataset/{}/{}'.format(platform, label)
        folders = os.listdir(path)
        for folder in folders:
            data = '{}/{}/'.format(path, folder)
            hasAnyFolder = False
            for files_in_folder in os.listdir(data):
                if files_in_folder == 'news content.json':
                    hasAnyFolder = True
                    with open('{}/{}'.format(data, files_in_folder)) as json_data:
                        json_data = json.load(json_data)
                        data_train = pd.DataFrame(pd.json_normalize(json_data))
                        labels.append(1 if label == 'fake' else 0)
                        for idx in range(data_train.shape[0]):
                            text = clean_str(data_train['text'][idx])
                            texts.append(text)
                            sentences = tokenize.sent_tokenize(text)
                            contents.append(sentences)
                            id = re.search(r"([0-9]+)", folder).group(1)
                            ids.append(id)
                            print(id)
                if files_in_folder == 'retweets':
                    for retweets in os.listdir('{}/{}'.format(data, files_in_folder)):
                        with open('{}/{}/{}'.format(data, files_in_folder, retweets)) as retweet_data:
                            for retweet in json.load(retweet_data)["retweets"]:
                                retweet_text = clean_str(retweet['text'])
                                tmp_comments.append(retweet_text)
            if hasAnyFolder:
                comments.append(tmp_comments)
            print(len(ids), len(labels), len(contents), len(comments))
    pickle.dump([ids, labels, contents, comments], open(file_name, "wb"))
    return ids, labels, contents, comments


#if __name__ == '__main__':
# dataset used for training
platform = 'politifact'
VALIDATION_SPLIT = 0.25
file_name = './dataset_defend.pkl'

if Path(file_name).is_file():
    print("if")
    ids, labels, contents, comments = pickle.load(open(file_name, "rb"))
else:
    print("else")
    ids, labels, contents, comments = get_dataset(file_name)

labels = np.asarray(labels)
labels = to_categorical(labels)

content_ids = set(ids)

id_train, id_test, x_train, x_val, y_train, y_val, c_train, c_val = train_test_split(ids, contents, labels, comments,
                                                                    test_size=VALIDATION_SPLIT, random_state=42,
                                                                    stratify=labels)

# Train and save the model
SAVED_MODEL_FILENAME = platform + '_Defend_model.h5'
h = Defend()
h.train(x_train, y_train, c_train, c_val, x_val, y_val,
        batch_size=20,
        epochs=30,
        embeddings_path='./glove.6B.100d.txt',
        saved_model_dir=SAVED_MODEL_DIR,
        saved_model_filename=SAVED_MODEL_FILENAME)

# h.load_weights(saved_model_dir = SAVED_MODEL_DIR, saved_model_filename = SAVED_MODEL_FILENAME)

# Get the attention weights for sentences in the news contents as well as comments
activation_maps = h.activation_maps(x_val, c_val)
