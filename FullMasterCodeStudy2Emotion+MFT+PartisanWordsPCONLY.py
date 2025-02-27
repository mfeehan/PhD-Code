import os
import ssl
import pandas as pd
import numpy as np
import nltk
from langdetect import detect, LangDetectException
from nrclex import NRCLex
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from transformers import pipeline, AutoTokenizer
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
import time

# Start timer at the very beginning
start_time = time.time()

# SSL context fix
ssl._create_default_https_context = ssl._create_unverified_context

# Download necessary NLTK resources if not already downloaded
nltk.download('punkt', quiet=True)
nltk.download('vader_lexicon', quiet=True)
nltk.download('stopwords', quiet=True)

# File paths
base_file_path = "C:\\Users\\feeha\\iCloudDrive\\PhD Research\\IRA\\BigDataFolder\\PoliticalUsers.csv"
final_output_path = "C:\\Users\\feeha\\iCloudDrive\\PhD Research\\IRA\\BigDataFolder\\PoliticalUsersProcessed.csv"  # ALWAYS use a new file name or it won't append correctly

# Expanded partisan keywords list
PARTISAN_KEYWORDS = [
    # General political terms
    'liberal', 'conservative', 'republican', 'democrat', 'right-wing', 'left-wing', 'progressive', 'alt-right',
    'establishment', 'deep state', 'drain the swamp', 'fake news', 'mainstream media', 'MSM',
    'Obama', 'Biden', 'Clinton', 'Trump', 'Pence', 'Sanders', 'Hillary', 'McConnell', 'Pelosi',
    'DNC', 'GOP', 'Republicans', 'Democrats', 'libertarian', 'independent',
    # 2009–2012 (Obama Presidency)
    'Obamacare', 'Tea Party', 'birth certificate', 'socialist', 'czar',
    # 2016 Election (Trump vs. Clinton)
    'Crooked Hillary', 'Lock her up', 'Make America Great Again', 'MAGA', 'Build the wall',
    'Trump Train', 'email server', 'Benghazi',
    # 2018 (Midterms and Trump Presidency)
    'Trump Derangement Syndrome', 'TDS', 'covfefe', 'alternative facts', 'deep state',
    # Social Movements
    'Hands up don\'t shoot', 'I can\'t breathe', 'Kaepernick', 'take a knee',
    'racial divide', 'Black Lives Matter', 'BLM', 'all lives matter', 'Blue Lives Matter',
    'antifa', 'white supremacy', 'race war', 'social justice', 'racism',
    # Climate and Environment
    'Paris Agreement', 'climate change hoax', 'EPA', 'fracking',
    # International Relations
    'Russia', 'Ukraine', 'Crimea', 'NATO', 'Putin', 'Syria', 'China', 'Iran', 'ISIS', 'terrorism',
    'collusion', 'Russia hoax', 'WikiLeaks', 'Podesta emails', 'al-Baghdadi', 'radical Islam', 'caliphate',
    'trade war', 'tariffs', 'intellectual property theft',
    # Media and Pop Culture
    'CNN sucks', 'Fox News', 'Hannity', 'Maddow', 'SJW', 'NPC', 'red pill', 'blue pill',
    # 2018 Events
    'Parkland shooting', 'March for Our Lives', 'Kavanaugh', 'Me Too', '#MeToo'
]

# Initialize the Sentiment Analyzer once
sia = SentimentIntensityAnalyzer()

# Define MFT models and columns
models = {
    "joshnguyen/mformer-loyalty": "Loyaltyclassification_result",
    "joshnguyen/mformer-sanctity": "Sanctityclassification_result",
    "joshnguyen/mformer-care": "Careclassification_result",
    "joshnguyen/mformer-fairness": "Fairnessclassification_result",
    "joshnguyen/mformer-authority": "Authorityclassification_result"
}

# Function to process sentiment and engagement on CPU
def process_sentiment_and_engagement(chunk):
    chunk = chunk.copy()  # Create a copy to avoid SettingWithCopyWarning

    # Ensure tweet_text is a string
    chunk['tweet_text'] = chunk['tweet_text'].astype(str)
    
    # Sentiment analysis using VADER
    chunk['sentiment_score'] = chunk['tweet_text'].apply(lambda x: sia.polarity_scores(x)['compound'])
    
    # Engagement calculation with log transformation
    chunk['engagement'] = chunk[['quote_count', 'reply_count', 'like_count', 'retweet_count']].sum(axis=1).apply(np.log1p)
    
    # Partisan word detection
    chunk['partisan_words'] = chunk['tweet_text'].str.lower().apply(
        lambda text: ', '.join([word for word in PARTISAN_KEYWORDS if word in text.split()]) or 'None'
    )
    
    # Affect analysis using NRCLex
    chunk['affect'] = chunk['tweet_text'].apply(lambda x: NRCLex(x).affect_frequencies)
    return chunk

# Function to calculate engagement per follower
def calculate_engagement_per_follower(chunk):
    chunk = chunk.copy()  # Create a copy to avoid SettingWithCopyWarning
    
    if 'follower_count' in chunk.columns:
        chunk['engagement_per_follower'] = chunk.apply(
            lambda row: row['engagement'] / row['follower_count'] if row['follower_count'] > 0 else np.nan,
            axis=1
        )
    else:
        print("Warning: 'follower_count' column is missing.")
        chunk['engagement_per_follower'] = np.nan
    return chunk

# GPU-based classification for all MFT models with increased batch size
def classify_mft_gpu_all(chunk, models, batch_size=8192):
    chunk = chunk.copy()  # Create a copy to avoid SettingWithCopyWarning
    
    for model_name, output_column in models.items():
        print(f"Processing model: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        pipe = pipeline("text-classification", model=model_name, tokenizer=tokenizer, device=0)
        
        text_data = chunk['tweet_text'].tolist()
        results = []
        
        for i in range(0, len(text_data), batch_size):
            batch = text_data[i:i + batch_size]
            outputs = pipe(batch, truncation=True, max_length=512)
            results.extend([f"{output['label']} ({output['score']:.2f})" for output in outputs])
        
        chunk[output_column] = results
    return chunk

# Function to process each chunk: filtering, sentiment, engagement, and GPU-based MFT classification
def process_and_classify_chunk(chunk, models):
    # Remove rows where 'tweet_text' is blank or NaN
    chunk = chunk[chunk['tweet_text'].notna() & (chunk['tweet_text'].str.strip() != '')]

    # Retain all original columns
    original_columns = chunk.columns.tolist()

    # Process sentiment and engagement, adding new columns
    chunk = process_sentiment_and_engagement(chunk)

    # Calculate engagement per follower
    chunk = calculate_engagement_per_follower(chunk)

    # GPU-based MFT classification
    chunk = classify_mft_gpu_all(chunk, models)

    # Ensure original columns are preserved in the output
    return chunk[original_columns + [
        'sentiment_score', 'engagement', 'partisan_words', 'affect', 'engagement_per_follower',
        "Loyaltyclassification_result", "Sanctityclassification_result", "Careclassification_result",
        "Fairnessclassification_result", "Authorityclassification_result"
    ]]


# Process the CSV file in chunks with parallel processing and collect log engagement values
log_engagement_values = []
chunk_size = 50000
with ThreadPoolExecutor() as executor:
    with pd.read_csv(base_file_path, chunksize=chunk_size) as reader:
        futures = []
        for chunk in reader:
            futures.append(executor.submit(process_and_classify_chunk, chunk, models))
        
        for future in futures:
            processed_chunk = future.result()
            log_engagement_values.extend(processed_chunk['engagement'].values)  # Collect log engagement values
            processed_chunk.to_csv(final_output_path, mode='a', header=not os.path.exists(final_output_path), index=False)

# Stop timer and print total runtime
end_time = time.time()
total_time = end_time - start_time
print(f"Data processing complete. Results saved to {final_output_path}")
print(f"Total runtime: {total_time:.2f} seconds")
