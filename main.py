# ---- Import Required Packages ----

# Stability API related imports
from imgurpython import ImgurClient
import textwrap
from PIL import Image, ImageDraw, ImageFont
import stability_sdk.interfaces.gooseai.generation.generation_pb2 as generation
from stability_sdk import client

# Image related imports
import io
from PIL import Image

# DALL-E related imports
from dalle_pytorch import OpenAIDiscreteVAE, DALLE

# JSON and data related imports
import json
from pandas import json_normalize
import pandas as pd
import re
import csv

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
import datetime

# Fetch top tweets from a news source


def fetch_news(news_source):
    # News source to fetch tweets from
    news_source = news_source

    # Fetch the 5 most recent tweets from the news source
    tweets = api.user_timeline(
        screen_name=news_source, count=300, tweet_mode="extended")

    # Extract relevant information from the tweets
    tweet_data = []

    # Current time
    now = datetime.datetime.now(datetime.timezone.utc)

    for tweet in tweets:
        # Check if tweet is within the past 24 hours
        tweet_age = now - tweet.created_at
        if tweet_age <= datetime.timedelta(days=1):
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

# Function to get Bible verse and keywords using GPT-3.5


def get_bible_verse_and_keywords(news):

    prompt = f"Given the following news: '{news}' - Provide EITHER a Bible verse or Bible parable that can provide hope or guidance. Explain the historical biblical background in 2-3 sentences, how it relates to the situation, and what we can learn from it. Please include a short quote or excerpt (2-3 sentences) from the verse or parable within the answer. Present in way to maximize engagement on Instagram, such line breaks, and relevant hashtags (only at end). Start each paragraph with a relevant emoji. Keep it under 250 words."

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an AI that can provide relevant bible verses and related context based on news."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=500,
        n=1,
        stop=None,
        temperature=0.7,
    )

    return response.choices[0]['message']['content']


# Function to trim the verse if it is too long
def trim_verse(verse):

    words = verse.split()
    trimmed = False

    while len(' '.join(words)) > 180:
        words.pop(0)
        trimmed = True

    if trimmed:
        trimmed_verse = '...' + ' '.join(words)
    else:
        trimmed_verse = verse

    return trimmed_verse


def get_verse_from_gpt_response(gpt_response):
    # Regex pattern to match the book and chapter of the Bible verse
    pattern_book = r'(\b[A-Za-z]+\s\d+:\d+\b)'
    match_book = re.findall(pattern_book, gpt_response)

    # Regex pattern to match the verse
    pattern_verse = r'"(.*?)[”|"]'
    match_verse = re.findall(pattern_verse, gpt_response)

    # Trim the verse if it is too long
    match_verse[0] = trim_verse(match_verse[0])

    final_verse = match_verse[0] + ' - ' + match_book[0]

    return final_verse


def get_image_prompt_from_verse(verse):
    prompt = f"Given the following bible verse: '{verse}' - provide 5 keywords to describe the verse separated by commas."

    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        max_tokens=100,
        n=1,
        stop=None,
        temperature=0.5,
    )

    return response.choices[0].text.strip()

# Function to generate image from text and save to local directory


def gen_image_from_text(verse_prompt):

    # Function to get Bible verse and keywords using GPT-3.5
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
                    "text": verse_prompt + ', sci-fi movie style, overcast, night sky, motion blur, blur, atmospheric lighting'
                }
            ],
            "cfg_scale": 7,
            "clip_guidance_preset": "FAST_BLUE",
            "height": 512,
            "width": 512,
            "samples": 1,
            "steps": 30,
        },
    )

    if response.status_code != 200:
        raise Exception("Non-200 response: " + str(response.text))

    data = response.json()

    for i, image in enumerate(data["artifacts"]):
        # with open(f"./out/v1_txt2img_{i}.png", "wb") as f:
        with open(f"./out/gen_image.png", "wb") as f:
            f.write(base64.b64decode(image["base64"]))


# ----- Image Editing -----


# Function to add text to image

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
        rectangle_fill = (255, 255, 255, 200)  # RGBA: white with 150/255 alpha
        draw.rectangle([x-5, y, x + text_width + 5, y +
                       font_size], fill=rectangle_fill)

        # Draw the text
        draw.text((x, y), line, font=font, fill=(0, 0, 0))  # Black text
        y += font_size + line_spacing  # Increase y with line_spacing

    # Save the new image
    image.save(output_path)


# ---- Host image on Imgur ----


def imgur_upload(client_id, image_path):
    headers = {
        'Authorization': 'Client-ID {}'.format(client_id),
    }
    url = 'https://api.imgur.com/3/image'
    with open(image_path, 'rb') as image_file:
        image_data = image_file.read()
    response = requests.post(url, headers=headers, data={'image': image_data})
    if response.status_code == 200:
        return json.loads(response.text)['data']['link']
    else:
        raise Exception('Failed to upload image to Imgur')

# ---- Post on Instagram ----


def publish_image(image_path, caption, imgur_client_id, insta_token, ig_user_id):

    imgur_client_id = imgur_client_id
    local_image_path = image_path

    access_token = insta_token
    ig_user_id = ig_user_id
    image_url = imgur_upload(imgur_client_id, local_image_path)

    post_url = 'https://graph.facebook.com/v16.0/{}/media'.format(ig_user_id)
    payload = {
        'image_url': image_url,
        'caption': caption,
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


# ----- Main -----

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
imgur_client_id = api_keys['imgur_client_id']
ig_user_id = api_keys['ig_user_id']

# Authenticate to the Twitter API
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)






def get_gpt_response(prompt):

    prompt = prompt

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are the social media account manager for an Instagram account that creates Biblical related posts. "},
            {"role": "user", "content": prompt},
        ],
        max_tokens=500,
        n=1,
        stop=None,
        temperature=0.5,
    )

    return response.choices[0]['message']['content']

prompt = """
Please select a quote from a book or sermon from one of the following pastors/authors: Timothy J Keller, Lee Strobel, 
Rick Warren, John Piper, David Platt, C.S. Lewis, A.W. Tozer, J.I. Packer, Francis Chan, Todd Burpo, Dane C. Ortlund, 
Jeff Vanderstelt. The quote should provide hope, guidance, inspiration, encouragement, or wisdom.

Please provide some text that can be utilized as the caption for an Instagram post. In the text, include the quote, author, source, context within the book/sermon, and how one can apply it in their lives/what one can learn from it. Present it in a way is engaging on social media. At the end of the text, include relevant hashtags that will maximize engagement. Keep it under 250 words. Keep the number of emojis used under 4 and a maximum of 1 emoji in a paragraph.
"""

caption = get_gpt_response(prompt)

prompt = f"present the key message from {caption} in 10 words of less. maximize for engagement on social media. it will be used to overlay ontop of a picture on instagram. don't include hashtags. it should be able to capture the audience's attention"

text = get_gpt_response(prompt)

prompt = f"Provide 5-10 words separated by commas that will generated a photo to depict the following passage: {caption}"
verse_prompt = get_gpt_response(prompt)
verse_prompt






def main():
    # Download and sort top tweets over past 24 hours,
    fetch_news('BBCWorld')  # BBCWorld, CNN, spectatorindex
    top_tweets = pd.read_csv('data/tweet_data.csv')
    twt = top_tweets['text'][0]  # Get top tweet

    twt = "The rapid advancements in AI, like ChatGPT, are increasingly threatening white-collar jobs, forcing individuals to find new ways to define self-worth beyond their careers. As higher-paid and creative jobs become more exposed to automation, the AI revolution will uniquely impact white-collar workers. Adapting to this new reality may involve focusing on skills and roles less susceptible to automation and seeking fulfillment in other aspects of life, such as relationships, hobbies, and personal growth."

    # Use tweet as prompt to generate Bible verse and caption
    gpt_response = get_bible_verse_and_keywords(twt)  # Get GPT response
    full_caption = twt + '\n\n - \n\n ' + gpt_response  # Create full caption

    # Parse verse from GPT response, append to csv of all generated verses
    bible_verse = get_verse_from_gpt_response(gpt_response)

    with open("data/verse.csv", mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([bible_verse])

    # Generate image from verse
    verse_prompt = get_image_prompt_from_verse(bible_verse)
    gen_image_from_text(verse_prompt)

    # Add bible verse to image
    image_path = "out/gen_image.png"
    bible_verse = bible_verse
    output_path = "out/gen_image_with_text.png"
    add_text_to_image(image_path, bible_verse, output_path)

    # Upload image with text from local to Imgur, then post to Insta
    image_path = 'out/gen_image_with_text.png'
    publish_image(image_path, full_caption,
                  imgur_client_id, insta_token, ig_user_id)


# ----------------
# Things to do:
# If verse is too long, add epilises to start of verse
# Add link to the news article in the caption
# don't repeat bible verses
