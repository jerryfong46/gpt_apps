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
        screen_name=news_source, count=300, tweet_mode="extended"
    )

    # Extract relevant information from the tweets
    tweet_data = []

    # Current time
    now = datetime.datetime.now(datetime.timezone.utc)

    for tweet in tweets:
        # Check if tweet is within the past 24 hours
        tweet_age = now - tweet.created_at
        if tweet_age <= datetime.timedelta(days=1):
            tweet_data.append(
                {
                    "id": tweet.id_str,
                    "text": tweet.full_text,
                    "retweets": tweet.retweet_count,
                    "likes": tweet.favorite_count,
                }
            )

    tweet_data_df = json_normalize(tweet_data)

    # Sort the tweets by engagement score (retweets, likes, and replies)
    tweet_data_df["engagement_score"] = tweet_data_df.apply(engagement_score, axis=1)
    tweet_data_df = tweet_data_df.sort_values(
        by="engagement_score", ascending=False
    ).reset_index(drop=True)

    # clean hyperlink text
    tweet_data_df["text"] = tweet_data_df["text"].apply(clean_hyperlinks)
    tweet_data_df.to_csv("data/tweet_data.csv", index=False)


# Clean the hyperlinks from the tweet text


def clean_hyperlinks(tweet):
    return re.sub(r"https://t.co/\w+", "", tweet)


# Calculate the engagement score of a tweet
def engagement_score(tweet):
    retweets_weight = 5
    likes_weight = 1
    return tweet["retweets"] * retweets_weight + tweet["likes"] * likes_weight


# Function to get Bible verse and keywords using GPT-3.5


def get_bible_verse_and_keywords(news):
    prompt = f"Given the following news: '{news}' - Provide EITHER a Bible verse or Bible parable that can provide hope or guidance. Explain the historical biblical background in 2-3 sentences, how it relates to the situation, and what we can learn from it. Please include a short quote or excerpt (2-3 sentences) from the verse or parable within the answer. Present in way to maximize engagement on Instagram, such line breaks, and relevant hashtags (only at end). Start each paragraph with a relevant emoji. Keep it under 250 words."

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "You are an AI that can provide relevant bible verses and related context based on news.",
            },
            {"role": "user", "content": prompt},
        ],
        max_tokens=500,
        n=1,
        stop=None,
        temperature=0.7,
    )

    return response.choices[0]["message"]["content"]


# Function to trim the verse if it is too long
def trim_verse(verse):
    words = verse.split()
    trimmed = False

    while len(" ".join(words)) > 180:
        words.pop(0)
        trimmed = True

    if trimmed:
        trimmed_verse = "..." + " ".join(words)
    else:
        trimmed_verse = verse

    return trimmed_verse


def get_verse_from_gpt_response(gpt_response):
    # Regex pattern to match the book and chapter of the Bible verse
    pattern_book = r"(\b[A-Za-z]+\s\d+:\d+\b)"
    match_book = re.findall(pattern_book, gpt_response)

    # Regex pattern to match the verse
    pattern_verse = r'"(.*?)[”|"]'
    match_verse = re.findall(pattern_verse, gpt_response)

    # Trim the verse if it is too long
    match_verse[0] = trim_verse(match_verse[0])

    final_verse = match_verse[0] + " - " + match_book[0]

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
    api_host = os.getenv("API_HOST", "https://api.stability.ai")
    api_key = api_keys["dream_studio"]

    if api_key is None:
        raise Exception("Missing Stability API key.")

    response = requests.post(
        f"{api_host}/v1/generation/{engine_id}/text-to-image",
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        json={
            "text_prompts": [
                {
                    "text": verse_prompt
                    + ", sci-fi movie style, overcast, night sky, motion blur, blur, atmospheric lighting"
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


# ----- DALL-E -----


def gen_image_from_text(verse_prompt):
    response = openai.Image.create(prompt=verse_prompt, n=1, size="1024x1024")
    image_url = response["data"][0]["url"]

    import requests

    # Send a HTTP request to the URL of the image
    response = requests.get(image_url)

    # Check that the request was successful
    if response.status_code == 200:
        # Open the url image, set stream to True, this will return the stream content
        response = requests.get(image_url, stream=True)

        # Open a local file with write-binary mode and write the image to the file
        with open("./out/gen_image.png", "wb") as out_file:
            out_file.write(response.content)
        del response


# ----- Image Editing -----


# Function to add text to image


def add_text_to_image(image_path, text, output_path):
    # Load the image
    image = Image.open(image_path)

    # Create a drawing object
    draw = ImageDraw.Draw(image, "RGBA")

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
        draw.rectangle(
            [x - 5, y, x + text_width + 5, y + font_size], fill=rectangle_fill
        )

        # Draw the text with shadow
        shadowcolor = "gray"
        draw.text((x - 1, y - 1), line, font=font, fill=shadowcolor)
        draw.text((x + 1, y - 1), line, font=font, fill=shadowcolor)
        draw.text((x - 1, y + 1), line, font=font, fill=shadowcolor)
        draw.text((x + 1, y + 1), line, font=font, fill=shadowcolor)

        # Draw the text
        draw.text((x, y), line, font=font, fill=(0, 0, 0))  # Black text
        y += font_size + line_spacing  # Increase y with line_spacing

    # Save the new image
    image.save(output_path)


# ---- Host image on Imgur ----


def imgur_upload(client_id, image_path):
    headers = {
        "Authorization": "Client-ID {}".format(client_id),
    }
    url = "https://api.imgur.com/3/image"
    with open(image_path, "rb") as image_file:
        image_data = image_file.read()
    response = requests.post(url, headers=headers, data={"image": image_data})
    if response.status_code == 200:
        return json.loads(response.text)["data"]["link"]
    else:
        raise Exception("Failed to upload image to Imgur")


# ---- Post on Instagram ----


def publish_image(image_path, caption, imgur_client_id, insta_token, ig_user_id):
    imgur_client_id = imgur_client_id
    local_image_path = image_path

    access_token = insta_token
    ig_user_id = ig_user_id
    image_url = imgur_upload(imgur_client_id, local_image_path)

    post_url = "https://graph.facebook.com/v16.0/{}/media".format(ig_user_id)
    payload = {"image_url": image_url, "caption": caption, "access_token": access_token}

    r = requests.post(post_url, data=payload)
    print(r.text)
    print("Media uploaded sucessfully")

    results = json.loads(r.text)
    if "id" in results:
        creation_id = results["id"]
        second_url = "https://graph.facebook.com/v16.0/{}/media_publish".format(
            ig_user_id
        )
        second_payload = {"creation_id": creation_id, "access_token": access_token}

        r = requests.post(second_url, data=second_payload)
        print(r.text)
        print("Media published to instagram")

    else:
        print("Media not published to instagram")


# ----- Main -----

# Define the path to the file containing API keys and descriptions
api_key_file = os.path.join("config", "api_keys.txt")

# Read the file and split the lines by newline
with open(api_key_file, "r") as f:
    api_key_lines = f.read().split("\n")

# Parse the lines into a dictionary
api_keys = {}
for line in api_key_lines:
    if line.strip():
        key_description, api_key = line.split()
        api_keys[key_description.strip("[]")] = api_key

# API Keys
consumer_key = api_keys["twitter_api_key"]
consumer_secret = api_keys["twitter_api_secret"]
access_token = api_keys["twitter_access_token"]
access_token_secret = api_keys["twitter_access_token_secret"]
openai.api_key = api_keys["openai"]
insta_token = api_keys["graph_api_access_token"]
imgur_client_id = api_keys["imgur_client_id"]
ig_user_id = api_keys["ig_user_id"]

# Authenticate to the Twitter API
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)


def get_gpt_response(prompt, model):
    prompt = prompt

    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are the social media account manager for an Instagram account that creates Biblical related posts. ",
            },
            {"role": "user", "content": prompt},
        ],
        max_tokens=500,
        n=1,
        stop=None,
        temperature=0.5,
    )

    return response.choices[0]["message"]["content"]


import random

human_experience = [
    "Loss of a Loved One",
    "Loss of a Friend",
    "Loss of a Family Member",
    "Loss of a Pet",
    "Falling in Love",
    "Heartbreak from a Romantic Relationship",
    "Heartbreak from a Broken Friendship",
    "Heartbreak from Family Conflict",
    "Fear of Failure at Work",
    "Fear of Failure at School",
    "Fear of Failure in a Personal Endeavor",
    "Achievement in Professional Life",
    "Achievement in Academic Life",
    "Achievement in Personal Goals",
    "Financial Stress from Debt",
    "Financial Stress from Job Loss",
    "Financial Stress from Poor Financial Decisions",
    "Birth of a Child",
    "Illness of a Family Member",
    "Illness of a Close Friend",
    "Feeling Alone in a Crowd",
    "Feeling Misunderstood by Friends",
    "Feeling Misunderstood by Family",
    "Maintaining a Long-term Friendship",
    "Ending a Friendship",
    "Moving to a New Country",
    "Starting a New Job",
    "Starting a New School",
    "Embarrassment in a Personal Situation",
    "Feeling Overwhelmed by Work or School",
    "Feeling Overwhelmed by Personal Responsibilities",
    "Feeling Overwhelmed by Relationships",
    "Dealing with Change in Personal Life",
    "Dealing with Change in Professional Life",
    "Dealing with Change in Health",
    "Ageing in Physical Health",
    "Ageing in Mental Health",
    "Struggle with Self-esteem",
    "Searching for Purpose in Work or Career",
    "Searching for Purpose in Relationships",
    "Searching for Purpose in Spiritual Life",
]

pastors_ls = [
    "Timothy J Keller",
    "Lee Strobel",
    "Rick Warren",
    "John Piper",
    "David Platt",
    "C.S. Lewis",
    "A.W. Tozer",
    "J.I. Packer",
    "Francis Chan",
    "Todd Burpo",
    "Dane C. Ortlund",
    "Jeff Vanderstelt",
]

medium_ls = ["book", "book", "book", "sermon"]

sel_experience = random.choice(human_experience)
sel_pastor = random.choice(pastors_ls)
sel_medium = random.choice(medium_ls)

prompt = f"""
You want to connect with the audience who might be experiencing: {sel_experience}.
Please select a quote from a {sel_medium} from the following pastor/author: {sel_pastor}. The quote should provide hope, guidance, inspiration, encouragement, or wisdom.

Please provide some text that can be utilized as the caption for an Instagram post. In the text, include the quote, author, source, context within the {sel_medium}, and how one can apply it in their lives/what one can learn from it. Present it in a way is engaging on social media. At the end of the text, include relevant hashtags that will maximize engagement. Keep it under 250 words. Keep the number of emojis used under 4 and a maximum of 1 emoji in a paragraph.
"""

caption = get_gpt_response(prompt, "gpt-4")

prompt = f"You are a social media expert. Present the key message from {caption} in 20 words of less. It should be concise enough to capture the audience's attention. it will be used to overlay ontop of a picture on instagram. it should be able to capture the audience's attention. Don't include hashtags or emojis."
text = get_gpt_response(prompt, "gpt-4")
text

prompt = f"You are a prompt engineering expert. Create a short metaphorical image description based on the themes : {text} The image description should be a maximum of 1 or 2 sentences and will be used by Dall-E to generate an image."
verse_prompt = get_gpt_response(prompt, "gpt-4")
verse_prompt

effects = (
    ", sci-fi movie style, overcast, night sky, motion blur, blur, atmospheric lighting"
)
gen_image_from_text(verse_prompt + effects)
# Add bible verse to image
image_path = "out/gen_image.png"
output_path = "out/gen_image_with_text.png"
add_text_to_image(image_path, text, output_path)

# Upload image with text from local to Imgur, then post to Insta
image_path = "out/gen_image_with_text.png"
publish_image(image_path, caption, imgur_client_id, insta_token, ig_user_id)


def main():
    # Download and sort top tweets over past 24 hours,
    fetch_news("BBCWorld")  # BBCWorld, CNN, spectatorindex
    top_tweets = pd.read_csv("data/tweet_data.csv")
    twt = top_tweets["text"][0]  # Get top tweet

    twt = "The rapid advancements in AI, like ChatGPT, are increasingly threatening white-collar jobs, forcing individuals to find new ways to define self-worth beyond their careers. As higher-paid and creative jobs become more exposed to automation, the AI revolution will uniquely impact white-collar workers. Adapting to this new reality may involve focusing on skills and roles less susceptible to automation and seeking fulfillment in other aspects of life, such as relationships, hobbies, and personal growth."

    # Use tweet as prompt to generate Bible verse and caption
    gpt_response = get_bible_verse_and_keywords(twt)  # Get GPT response
    full_caption = twt + "\n\n - \n\n " + gpt_response  # Create full caption

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
    image_path = "out/gen_image_with_text.png"
    publish_image(image_path, full_caption, imgur_client_id, insta_token, ig_user_id)


# ----------------
# Things to do:
# If verse is too long, add epilises to start of verse
# Add link to the news article in the caption
# don't repeat bible verses






class NewsRetriever:
    def __init__(self, twitter_api):
        # initialize with necessary API keys or client
        self.api = twitter_api

    def get_top_news(self):
        # implement logic to fetch and return top news


class ExperiencePicker:
    def __init__(self, experiences, authors):
        self.experiences = experiences
        self.authors = authors

    def pick_experience_and_author(self):
        # implement logic to pick and return a random experience and author


class ContentGenerator:
    def __init__(self, openai_api):
        # initialize with necessary API keys or client
        self.api = openai_api

    def generate_caption(self, prompt):
        # implement logic to generate and return a caption based on the prompt


class ImageManager:
    def __init__(self, openai_api):
        # initialize with necessary API keys or client
        self.api = openai_api

    def generate_image(self, verse_prompt):
        # implement logic to generate an image based on the verse_prompt

    def add_text_to_image(self, image_path, text):
        # implement logic to add text to an image


class PostPublisher:
    def __init__(self, imgur_client_id, insta_token, ig_user_id):
        # initialize with necessary parameters
        self.imgur_client_id = imgur_client_id
        self.insta_token = insta_token
        self.ig_user_id = ig_user_id

    def publish_image(self, image_path, caption):
        # implement logic to publish an image with caption on Instagram


class SocialMediaManager:
    def __init__(self, twitter_api, openai_api, imgur_client_id, insta_token, ig_user_id, experiences, authors):
        self.news_retriever = NewsRetriever(twitter_api)
        self.experience_picker = ExperiencePicker(experiences, authors)
        self.content_generator = ContentGenerator(openai_api)
        self.image_manager = ImageManager(openai_api)
        self.post_publisher = PostPublisher(imgur_client_id, insta_token, ig_user_id)

    def create_news_post(self):
        # 1. Get top news
        # 2. Generate caption
        # 3. Generate image

class NewsBasedPost:
    def __init__(self):
        self.news_retriever = NewsRetriever()
        self.verse_finder = VerseFinder()
        self.image_generator = ImageGenerator()
        self.content_creator = ContentCreator()
        self.post_creator = InstagramPostCreator()

    def create_post(self):
        news = self.news_retriever.get_top_news()
        verse = self.verse_finder.find_related_verse(news)
        image = self.image_generator.generate_image(news, verse)
        content = self.content_creator.create_content(news, verse)
        self.post_creator.create_post(image, content)


class ExperienceBasedPost:
    def __init__(self):
        self.experience_picker = ExperiencePicker()
        self.author_picker = AuthorPicker()
        self.book_picker = BookPicker()
        self.image_generator = ImageGenerator()
        self.content_creator = ContentCreator()
        self.post_creator = InstagramPostCreator()

    def create_post(self):
        experience = self.experience_picker.pick_random_experience()
        author = self.author_picker.pick_random_author()
        book = self.book_picker.pick_book(author)
        image = self.image_generator.generate_image(experience, author, book)
        content = self.content_creator.create_content(experience, author, book)
        self.post_creator.create_post(image, content)

