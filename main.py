import json
import tweepy
import os
from pandas import json_normalize
import re

# Define the path to the file containing API keys and descriptions
api_key_file = os.path.join('config', 'api_keys.txt')

# Read the file and split the lines by newline
with open(api_key_file, 'r') as f:
    api_key_lines = f.read().split('\n')

# Parse the lines into a dictionary
api_keys = {}
for line in api_key_lines:
    if line.strip():
        key_description, api_key = line.split()
        api_keys[key_description.strip('[]')] = api_key

# Print the API keys
api_keys['twitter_api_key']
api_keys['twitter_api_secret']
api_keys['twitter_bearer_token']
api_keys['twitter_access_token']
api_keys['twitter_access_token_secret']


# Replace these with your own Twitter API credentials
consumer_key = api_keys['twitter_api_key']
consumer_secret = api_keys['twitter_api_secret']
access_token = api_keys['twitter_access_token']
access_token_secret = api_keys['twitter_access_token_secret']

# Authenticate to the Twitter API
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)


# Fetch top 3 tweets from a news source
def fetch_news(news_source):
    # News source to fetch tweets from
    news_source = news_source

    # Fetch the 5 most recent tweets from the news source
    tweets = api.user_timeline(
        screen_name=news_source, count=5, tweet_mode="extended")

    # Extract relevant information from the tweets
    tweet_data = []
    for tweet in tweets:
        tweet_data.append({
            "id": tweet.id_str,
            "text": tweet.full_text,
            "retweets": tweet.retweet_count,
            "likes": tweet.favorite_count
        })

    tweet_data_df = json_normalize(tweet_data)

    # Sort the tweets by engagement score (retweets, likes, and replies)
    tweet_data_df['engagement_score'] = tweet_data_df.apply(
        engagement_score, axis=1)
    tweet_data_df = tweet_data_df.sort_values(
        by='engagement_score', ascending=False).reset_index(drop=True)
    tweet_text = '|'.join(tweet_data_df['text'][0:3])
    tweet_text = re.sub(r'https://t.co/\w+', '', tweet_text)

    return tweet_text


# Calculate the engagement score of a tweet


def engagement_score(tweet):
    retweets_weight = 5
    likes_weight = 1

    return (tweet["retweets"] * retweets_weight + tweet["likes"] * likes_weight)


fetch_news('BBCWorld')
