"""
HCA Emotion Analysis - Laptop Version  
Quick test of transformer models on fairy tale text
"""

import numpy as np
import pandas as pd
import re
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification, Trainer
from collections import defaultdict
from tqdm import tqdm
import accelerate
import datetime

# For sentence tokenization
import nltk
# nltk.download('punkt') # Download once
# nltk.download('punkt_tab')  # Download once
from nltk.tokenize import sent_tokenize 

# Planned models:   distilroberta-base (https://huggingface.co/j-hartmann/emotion-english-distilroberta-base)
#                   SamLowe/roberta-base-go_emotions   
#                   cardiffnlp/twitter-roberta-base-sentiment-latest

tales = pd.read_csv("Data/hca_tales.csv")
pred_texts = tales['text'].tolist()

### Adding in own code to chunk text appropriately for model input length limits
def chunk_text(df=tales, title_col='tale', text_col='text', max_length=510):
    """
    Chunk text into segments of max_length tokens.

    Input:
     - df: a dataframe which has at least a column for the title of the text you wish to analyze, and a column for the text itself
     - text_col: column name for the text column
     - title_col: column name for the title of the text
     - max_length: maximum tokens allowed. uses tokenizer.tokenize() to determine sizes
    
    General idea: 
    
    Split text into sentences, and scan sentence-by-sentence until the tokens reach max length (512 is max length for model, we will cut at 510 to save room for CLS and SEP (start & end) tokens). Ideally, we make cutoffs end at sentence endpoints. 

    - useful to have # sentences/chunk?
    """

    results = []

    print('Beginning analysis at {datetime.datetime.now()}.')

    for idx, row in df.iterrows():
        title = row[title_col]
        text = row[text_col]

        print(f"On tale {title}")

        sentences = nltk.sent_tokenize(text)

        current_tokens = 0
        current_chunk = ""
        chunk_number = 0
        sentence_count_in_chunk = 0

        for sent in sentences:
            sent = sent.strip()
            if not sent: 
                continue # if nonexistent sentence after stripping, skip
            
            sentence_tokens = len(tokenizer.tokenize(sent))
            
            # Check if adding this sentence would exceed max_length
            if current_tokens + sentence_tokens > max_length:
                
                if current_chunk: # chunk exists, save it
                    results.append({
                        'tale': title,
                        'chunk': chunk_number,
                        'text': current_chunk.strip(),
                        'token_count': current_tokens,
                        'sentence_count': sentence_count_in_chunk
                    })
                    
                    # Start new chunk with current sentence
                    chunk_number += 1
                    current_chunk = sent
                    current_tokens = sentence_tokens
                    sentence_count_in_chunk = 1
                    
                else: # Single sentence longer than max_length
                    print(f"Warning: Sentence longer than max length. Max: {max_length}, sentence: {sentence_tokens}")
                    # Save it as its own chunk anyway
                    results.append({
                        'tale': title,
                        'chunk': chunk_number,
                        'text': sent,
                        'token_count': sentence_tokens,
                        'sentence_count': 1
                    })
                    chunk_number += 1
                    current_chunk = ""
                    current_tokens = 0
                    sentence_count_in_chunk = 0
                
            else: # Add sentence to current chunk
                if current_chunk: 
                    current_chunk = current_chunk + " " + sent
                else: 
                    current_chunk = sent
                current_tokens += sentence_tokens
                sentence_count_in_chunk += 1

        # Save final chunk if it exists
        if current_chunk: 
            results.append({
                'tale': title,
                'chunk': chunk_number,
                'text': current_chunk.strip(),
                'token_count': current_tokens,
                'sentence_count': sentence_count_in_chunk
            })
    
    # Convert to DataFrame
    chunks_df = pd.DataFrame(results)
    
    print(f'\nCompleted analysis at {datetime.datetime.now()}.')
    print(f'Created {len(chunks_df)} total chunks from {len(df)} tales.')
    
    return chunks_df
###

#####
## From the distilroberta documentation; provides a workflow for analyzing sets of texts
# Create class for data preparation
class SimpleDataset:
    def __init__(self, tokenized_texts):
        self.tokenized_texts = tokenized_texts
    
    def __len__(self):
        return len(self.tokenized_texts["input_ids"])
    
    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.tokenized_texts.items()}

# load tokenizer and model, create trainer
model_name = "j-hartmann/emotion-english-distilroberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
trainer = Trainer(model=model)

# create list of texts (can be imported from .csv, .xls etc.)
# pred_texts = ['I like that', 'That is annoying', 'This is great!', 'Wouldn´t recommend it.']

chunks_df = chunk_text(tales, 'book', 'text')

# Display results
print(f"Dataframe shape: {chunks_df.shape}")
print(f"Columns: {list(chunks_df.columns)}")
print(chunks_df.head())

def analyze_emotions(df, text_col='text', model_name="j-hartmann/emotion-english-distilroberta-base"):
    """
    Add emotion analysis to a chunked dataframe.
    
    Arguments:
        df: Chunked dataframe with columns like 'tale', 'chunk', 'text', etc.
        text_col: Text column of dataframe
        model_name: HuggingFace model
    
    Returns:
        DataFrame with original columns plus emotion analysis columns
    """
    print(f'Beginning emotion analysis at {datetime.datetime.now()}.')
    print(f'Processing {len(df)} chunks with model: {model_name}')
    
    # Initialize the classifier
    classifier = pipeline("text-classification", 
                         model=model_name, 
                         top_k=None)
    
    # Get emotion labels from sample prediction
    sample_result = classifier("Sample text to get emotion labels.")[0]
    emotion_labels = [emotion['label'] for emotion in sample_result]
    
    print(f'Model predicts {len(emotion_labels)} emotions: {emotion_labels}')
    
    # Copy the original dataframe
    result_df = df.copy()
    
    # Get all emotion predictions at once
    texts = df[text_col].tolist()
    all_results = classifier(texts)
    
    # Initialize emotion score columns
    for emotion in emotion_labels:
        result_df[f'{emotion}_score'] = 0.0
    
    # Initialize derived metric columns
    # result_df['dominant_emotion'] = ''
    # result_df['dominant_score'] = 0.0
    # result_df['second_emotion'] = ''
    # result_df['second_score'] = 0.0
    # result_df['confidence_gap'] = 0.0
    # result_df['uncertainty'] = 0.0
    # result_df['emotion_entropy'] = 0.0
    
    # # Add derived psychological metrics if standard emotions are available
    # has_standard_emotions = all(emotion in emotion_labels for emotion in ['joy', 'anger', 'sadness', 'fear'])
    # has_neutral = 'neutral' in emotion_labels
    
    # if has_standard_emotions:
    #     result_df['polarity_score'] = 0.0
    #     result_df['arousal_score'] = 0.0
    #     result_df['emotional_intensity'] = 0.0
    #     result_df['emotion_complexity'] = 0.0
    
    # Process each prediction
    for i, emotions_result in enumerate(all_results):
        # Extract individual emotion scores
        emotion_scores = {emotion['label']: emotion['score'] for emotion in emotions_result}
        
        # Set emotion score columns
        for emotion in emotion_labels:
            result_df.loc[i, f'{emotion}_score'] = emotion_scores[emotion]
        
        # Calculate derived metrics
        sorted_emotions = sorted(emotions_result, key=lambda x: x['score'], reverse=True)
        # dominant = sorted_emotions[0]
        # second = sorted_emotions[1]
        
        # result_df.loc[i, 'dominant_emotion'] = dominant['label']
        # result_df.loc[i, 'dominant_score'] = dominant['score']
        # result_df.loc[i, 'second_emotion'] = second['label']
        # result_df.loc[i, 'second_score'] = second['score']
        # result_df.loc[i, 'confidence_gap'] = dominant['score'] - second['score']
        # result_df.loc[i, 'uncertainty'] = 1 - dominant['score']
        
        # # Calculate entropy (measure of emotion distribution uncertainty)
        # entropy = -sum(emotion['score'] * np.log(emotion['score'] + 1e-10) for emotion in emotions_result)
        # result_df.loc[i, 'emotion_entropy'] = entropy
        
        # # Calculate psychological metrics for standard emotion models
        # if has_standard_emotions:
        #     # Polarity (positive vs negative valence)
        #     positive = emotion_scores.get('joy', 0) + emotion_scores.get('surprise', 0)
        #     negative = (emotion_scores.get('anger', 0) + emotion_scores.get('sadness', 0) + 
        #                emotion_scores.get('fear', 0) + emotion_scores.get('disgust', 0))
        #     result_df.loc[i, 'polarity_score'] = positive - negative
            
        #     # Arousal (high vs low energy)
        #     high_arousal = (emotion_scores.get('anger', 0) + emotion_scores.get('fear', 0) + 
        #                    emotion_scores.get('surprise', 0) + emotion_scores.get('joy', 0))
        #     low_arousal = (emotion_scores.get('sadness', 0) + emotion_scores.get('disgust', 0) + 
        #                   emotion_scores.get('neutral', 0))
        #     result_df.loc[i, 'arousal_score'] = high_arousal - low_arousal
            
        #     # Emotional intensity (non-neutral emotions)
        #     if has_neutral:
        #         result_df.loc[i, 'emotional_intensity'] = 1 - emotion_scores['neutral']
        #     else:
        #         result_df.loc[i, 'emotional_intensity'] = 1 - min(emotion_scores.values())
            
        #     # Emotional complexity (distribution evenness)
        #     scores = [emotion['score'] for emotion in emotions_result]
        #     complexity = 1 - sum(score**2 for score in scores)  # 1 - sum of squares
        #     result_df.loc[i, 'emotion_complexity'] = complexity
    
    print(f'Completed emotion analysis at {datetime.datetime.now()}.')
    
    # Count added columns
    basic_emotion_cols = len(emotion_labels)
    # derived_cols = 7 + (4 if has_standard_emotions else 0)
    print(f'Added {basic_emotion_cols} emotion scores.') #+ {derived_cols} derived metrics.')
    
    return result_df

# Usage:
emotion_df = analyze_emotions(chunks_df)

# Display results
print("New columns added:")
new_cols = [col for col in emotion_df.columns if col not in chunks_df.columns]
print(new_cols)

# Export to csv
emotion_results_file = "emotion_results.csv"
emotion_df.to_csv(emotion_results_file, index=False)

###

models = [
    'j-hartmann/emotion-english-distilroberta-base',
    'cardiffnlp/twitter-roberta-base-sentiment-latest',
    'SamLowe/roberta-base-go_emotions'
]

# Full tale texts do not fit in the token max lengths of the models for many of the tales. Find max token length for each model, we'll segment stories into overlapping sequences of these lengths.
t1 = AutoTokenizer.from_pretrained('j-hartmann/emotion-english-distilroberta-base').model_max_length
t2 = AutoTokenizer.from_pretrained('cardiffnlp/twitter-roberta-base-sentiment-latest').model_max_length
t3 = AutoTokenizer.from_pretrained('SamLowe/roberta-base-go_emotions').model_max_length
# Is there a more succinct way to write the above finding of the minimum token length across multiple models, and which model it is which is limiting this size?

