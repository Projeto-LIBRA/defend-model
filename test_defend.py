# coding=utf-8
from __future__ import unicode_literals
from datetime import datetime
import json
import boto3
import os
from urllib.request import urlopen, Request
from pathlib import Path
from defend import Defend
import pandas as pd
import tweepy
import re
import numpy as np
#from sklearn.model_selection import train_test_split
#from nltk import tokenize
#from keras.utils.np_utils import to_categorical
import pickle

FEW_FOLLOWERS_THRESHOLD = 20

consumer_key = "",
consumer_secret = "",
access_token="",
access_token_secret=""

s3 = boto3.resource('s3')
raw_bucket = 'collected-tweets'
processed_bucket = 'model-processing-results'

def clean_str(string):
    """
    Tokenization/string cleaning for dataset
    Every dataset is lower cased except
    """
    string = re.sub(r"\\", "", string)
    string = re.sub(r"\'", "", string)
    string = re.sub(r"\"", "", string)
    return string.strip().lower()

def read_s3_json(s3_path):
    print('reading {}'.format(s3_path))
    s3_obj = s3.Object(raw_bucket, s3_path)
    s3_json = json.loads(s3_obj.get()['Body'].read().decode('utf-8'))
    print('succsessfuly read {}'.format(s3_path))
    return s3_json


def write_to_s3(data, s3_path, file_name):
    s3_object = s3.Object(processed_bucket, s3_path + file_name + '.json')
    s3_object.put(Body=bytes(json.dumps(data).encode('UTF-8')), ContentType='application/json')
    print("Wrote {}.json to {}".format(file_name, processed_bucket))

seems_fake = 'Cuidado, o tweet e seus retweets possuem padrões linguísticos suspeitos!\n'
seems_ok = 'O tweet e seus retweets não possuem padrões linguísticos suspeitos!\n'

more_info = '\nAlgumas informações:\n'
same_text_tweets = '-{} tweets com o mesmo texto foram feitos no último mês\n'
few_followers_retweets = '-{} usuários com menos de {} seguidores retuitaram o post\n'
rate_reply = '\nMe avalie: bit.ly/AvalieLibra'
know_more = '\nAcesse nosso perfil para saber mais!'

def get_reply_text(is_fake, same_text_amount, few_followers_amount):
    additional_info = more_info + same_text_tweets.format(same_text_amount) + few_followers_retweets.format(few_followers_amount, FEW_FOLLOWERS_THRESHOLD) + rate_reply + know_more
    if is_fake:
        return seems_fake + additional_info
    else:
        return seems_ok + additional_info


def lambda_handler(event, context):
    # Get tweet and tag ids from event
    tweet_info = json.loads(event['Records'][0]['body'])
    tweet_id = tweet_info['conversation_id']
    tag_id = tweet_info['tag_id']
 
    # Train and save the model
    SAVED_MODEL_DIR = 'saved_models'
    SAVED_MODEL_FILENAME = 'politifact_Defend_model.h5' 
    h = Defend()
    h.load_weights(saved_model_dir = SAVED_MODEL_DIR, saved_model_filename = SAVED_MODEL_FILENAME)
 
    # Read tweet data from s3
    tweet_data_path = '{}/{}/{}'.format(tweet_id, tag_id, '{}.json'.format(tweet_id))
    tweet_json = read_s3_json(tweet_data_path)
    tweet_text = clean_str(tweet_json['full_text'])
    
    # Read retweets data from s3
    retweets_data_path = '{}/{}/{}'.format(tweet_id, tag_id, 'retweets.json')
    retweets_json = read_s3_json(retweets_data_path)
    retweets_list = []
    for retweet in retweets_json['retweets']:
        retweet_text = clean_str(retweet['full_text'])
        retweets_list.append(retweet_text)

    # Read replies data from s3
    replies_data_path = '{}/{}/{}'.format(tweet_id, tag_id, 'replies.json')
    replies_json = read_s3_json(replies_data_path)
    replies_list = []
    for reply in replies_json['replies']:
        reply_text = clean_str(reply['full_text'])
        replies_list.append(reply_text)

    # Read quote_tweets data from s3
    quote_tweets_data_path = '{}/{}/{}'.format(tweet_id, tag_id, 'quote_tweets.json')
    quote_tweets_json = read_s3_json(quote_tweets_data_path)
    quote_tweets_list = []
    for quote_tweet in quote_tweets_json['quote_tweets']:
        quote_tweet_text = clean_str(quote_tweet['text'])
        quote_tweets_list.append(quote_tweet_text)
    
    # texto do tweet
    x = [[tweet_text]]

    # textos dos replies e dos quote_tweets
    c = [quote_tweets_list + replies_list]

    # Predict
    val_predict_onehot = (np.asarray(h.predict(x, c))).round()
    val_predict = np.argmax(val_predict_onehot, axis=1)

    # activation_maps = h.activation_maps(x, c)

    # print('val_predict_onehot:')
    # print(val_predict_onehot)
    
    # print('val_predict:')
    # print(val_predict)

    is_fake = bool(val_predict[0])

    few_followers_retweets_amount = len(list(filter(lambda retweet: retweet['user']['followers_count'] < FEW_FOLLOWERS_THRESHOLD, retweets_json['retweets'])))

    same_text_tweets_path = '{}/{}/{}'.format(tweet_id, tag_id, 'same_text_tweets.json')
    same_text_tweets_json = read_s3_json(same_text_tweets_path)
    same_text_tweets_amount = len(same_text_tweets_json['same_text_tweets'])

    reply_text = get_reply_text(is_fake, same_text_tweets_amount, few_followers_retweets_amount)
    
    reply_tweet(reply_text, tweet_info)

    save_result(reply_text, tweet_info, tweet_text, retweets_list, replies_list, quote_tweets_list)


def save_result(reply_text, tweet_info, tweet_text, retweets_texts, replies_texts, quote_tweets_texts):
    result_json = tweet_info
    result_json['reply_text'] = reply_text
    result_json['tweet_text'] = tweet_text
    result_json['retweets_texts'] = retweets_texts
    result_json['replies_texts'] = replies_texts
    result_json['quote_tweets_texts'] = quote_tweets_texts
    result_json['result_created_at'] = str(datetime.now()).replace(' ', '_')
    write_to_s3(result_json, '{}/{}/'.format(tweet_info['conversation_id'], tweet_info['tag_id']), 'result')
    
def reply_tweet(text, tweet_info):
    tag_id = tweet_info['tag_id']
    tag_created_at = tweet_info['tag_created_at']
    conversation_id = tweet_info['conversation_id']

    client = tweepy.Client(consumer_key = consumer_key,
                        consumer_secret = consumer_secret,
                        access_token=access_token,
                        access_token_secret = access_token_secret)

    # Replace the text with whatever you want to Tweet about
    response = client.create_tweet(text=text, in_reply_to_tweet_id=int(tag_id))

    print(response)
