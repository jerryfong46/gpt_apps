# Stability API related imports
import textwrap
from PIL import Image, ImageDraw, ImageFont
import stability_sdk.interfaces.gooseai.generation.generation_pb2 as generation
from stability_sdk import client

# Image related imports
from IPython.display import display
import io
from PIL import Image

# DALL-E related imports
from dalle_pytorch import OpenAIDiscreteVAE, DALLE

# JSON and data related imports
import json
from pandas import json_normalize
import pandas as pd
import re

# Twitter related imports
import tweepy

# OpenAI related imports
import openai

# Other miscellaneous imports
import warnings
import getpass
import os
import base64
import requests


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

# API Keys
consumer_key = api_keys['twitter_api_key']
consumer_secret = api_keys['twitter_api_secret']
access_token = api_keys['twitter_access_token']
access_token_secret = api_keys['twitter_access_token_secret']
openai.api_key = api_keys['openai']
insta_token = api_keys['graph_api_access_token']


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

    # clean hyperlink text
    tweet_data_df['text'] = tweet_data_df['text'].apply(clean_hyperlinks)
    tweet_data_df.to_csv('data/tweet_data.csv', index=False)

# Clean the hyperlinks from the tweet text


def clean_hyperlinks(tweet):
    return re.sub(r'https://t.co/\w+', '', tweet)


# Calculate the engagement score of a tweet


def engagement_score(tweet):
    retweets_weight = 5
    likes_weight = 1

    return (tweet["retweets"] * retweets_weight + tweet["likes"] * likes_weight)


fetch_news('BBCWorld')


# Function to get Bible verse and keywords using GPT-3.5


def get_bible_verse_and_keywords(news):
    prompt = f"Given the following news: '{news}' - Provide a relevant bible verse that might provide reassurance, encouragement, motivation, inspiration, guidance, comfort, spiritual growth or wisdom. First provide context for the event, then present the verse, and then highlight how the verse relates to it. Example: In the wake of the recent tragedy in the French Alps, we turn to Psalm 46:1 for comfort and reassurance: 'God is our refuge and strength, an ever-present help in trouble."

    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        max_tokens=150,
        n=1,
        stop=None,
        temperature=0.5,
    )

    return response.choices[0].text.strip()


# Read in top tweets to create verse
top_tweets = pd.read_csv('data/tweet_data.csv')

twt = top_tweets['text'][1]
get_bible_verse_and_keywords(twt)


api_keys['dream_studio']


engine_id = "stable-diffusion-v1-5"
api_host = os.getenv('API_HOST', 'https://api.stability.ai')
api_key = api_keys['dream_studio']

if api_key is None:
    raise Exception("Missing Stability API key.")

response = requests.post(
    f"{api_host}/v1/generation/{engine_id}/text-to-image",
    headers={
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": f"Bearer {api_key}"
    },
    json={
        "text_prompts": [
            {
                "text": "A lighthouse on a cliff"
            }
        ],
        "cfg_scale": 7,
        "clip_guidance_preset": "FAST_BLUE",
        "height": 512,
        "width": 512,
        "samples": 1,
        "steps": 10,
    },
)

if response.status_code != 200:
    raise Exception("Non-200 response: " + str(response.text))

data = response.json()

for i, image in enumerate(data["artifacts"]):
    with open(f"./out/v1_txt2img_{i}.png", "wb") as f:
        f.write(base64.b64decode(image["base64"]))


# ----- Image Editing -----


def add_text_to_image(image_path, text, output_path):
    # Load the image
    image = Image.open(image_path)

    # Create a drawing object
    draw = ImageDraw.Draw(image, 'RGBA')

    # Define font properties (adjust the path to your desired font and its size)
    font_path = "font/Helvetica.ttf"
    font_size = 30
    font = ImageFont.truetype(font_path, font_size)

    # Set the margins
    margin_x = 20
    margin_y = 20

    # Wrap the text
    max_width = image.width - (2 * margin_x)
    avg_char_width = font.getlength("a")
    chars_per_line = max_width // avg_char_width
    lines = textwrap.wrap(text, width=chars_per_line)

    # Calculate the initial text position (vertically centered)
    total_text_height = len(lines) * font_size
    y = (image.height - total_text_height) // 2

    # Define line spacing
    line_spacing = 5

    # Draw the wrapped text on the image
    for line in lines:
        text_width = draw.textlength(line, font)
        x = margin_x + (max_width - text_width) // 2

        # Draw the translucent white rectangle behind the text
        rectangle_fill = (255, 255, 255, 150)  # RGBA: white with 150/255 alpha
        draw.rectangle([x-5, y, x + text_width + 5, y +
                       font_size], fill=rectangle_fill)

        # Draw the text
        draw.text((x, y), line, font=font, fill=(0, 0, 0))  # Black text
        y += font_size + line_spacing  # Increase y with line_spacing

    # Save the new image
    image.save(output_path)


# Example usage
image_path = "out/your_image.png"
bible_verse = "John 3:16 - For God so loved the world, that he gave his only begotten Son, that whosoever believeth in him should not perish, but have everlasting life."
output_path = "out/your_image_with_text.png"

add_text_to_image(image_path, bible_verse, output_path)


# ---- Post on Instagram ----

def publish_image():
    access_token = insta_token
    ig_user_id = '17841458791296115'
    image_url = 'https://i.swncdn.com/media/1600w/cms/BST/45821-inspirational%20verses.800w.tn.webp'

    post_url = 'https://graph.facebook.com/v16.0/{}/media'.format(ig_user_id)
    payload = {
        'image_url': image_url,
        'caption': 'This is a test caption',
        'access_token': access_token
    }

    r = requests.post(post_url, data=payload)
    print(r.text)
    print('Media uploaded sucessfully')

    results = json.loads(r.text)
    if 'id' in results:
        creation_id = results['id']
        second_url = 'https://graph.facebook.com/v16.0/{}/media_publish'.format(
            ig_user_id)
        second_payload = {
            'creation_id': creation_id,
            'access_token': access_token
        }

        r = requests.post(second_url, data=second_payload)
        print(r.text)
        print('Media published to instagram')
    else:
        print('Media not published to instagram')


publish_image()
