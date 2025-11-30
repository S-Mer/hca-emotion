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

# Planned models:   distilroberta-base (https://huggingface.co/j-hartmann/emotion-english-distilroberta-base)
#                   SamLowe/roberta-base-go_emotions   
#                   cardiffnlp/twitter-roberta-base-sentiment-latest

# Outline test function
def test_emotion_analysis():
    print("Testing with distilroberta-base")
    
    try:
        # Load emotion model
        print("Loading emotion model...")
        emotion_model = pipeline(
            'text-classification',
            model='j-hartmann/emotion-english-distilroberta-base',
            return_all_scores=True,
            device=-1,
            framework='pt' 
        )
        
        # Test with fairy tale text
        test_text = "It was lovely summer weather in the country, and the golden corn, the green oats, and the haystacks piled up in the meadows looked beautiful."
        
        print(f"Analyzing: '{test_text[:50]}...'")
        
        results = emotion_model(test_text)[0]
        
        print("\nEmotion Results:")
        for emotion in sorted(results, key=lambda x: x['score'], reverse=True):
            print(f"   {emotion['label']}: {emotion['score']:.3f}")
        
        print("\nAnalysis done!")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    test_emotion_analysis()

## Beginning analysis of tales with multiple models

# Import HCA tales dataset
tales = pd.read_csv("Data/hca_tales.csv")

tales.head()


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

token_lengths = {
    'distilroberta': AutoTokenizer.from_pretrained('j-hartmann/emotion-english-distilroberta-base').model_max_length,
    'twitter_roberta': AutoTokenizer.from_pretrained('cardiffnlp/twitter-roberta-base-sentiment-latest').model_max_length,
    'go_emotions': AutoTokenizer.from_pretrained('SamLowe/roberta-base-go_emotions').model_max_length
}
min_model = min(token_lengths, key=token_lengths.get)
min_length = token_lengths[min_model]
print(f"Max token lengths by model: {token_lengths}")
print(f"Model with minimum max token length: {min_model} ({min_length})")
# How do i find the documentation for these models, where you can see that .model_max_length is a property of the tokenizer class? i'm asking you, copilot. 3

print(f"Max token length across models: {min(t1, t2, t3)}")
# Is there a more succinct way to write the above finding of the minimum token length across multiple models, and which model it is which is limiting this size?


# Models require different tokenization approaches or chunk sizes; define a common tokenization across models. Given the relatively small size of the dataset (<200 tales, shorter stories in general), we can likely chunk tales in sliding windows with a set overlap to ensure coverage (we'll use max 510 as a default; the lowest is distilROBERTa at 512).
def chunk_text_with_overlap(text, tokenizer, max_tokens=510, overlap_sentences=2):
    """
    Chunk text respecting sentence boundaries with sentence-level overlap.
    
    Args:
        text: Full story text
        tokenizer: HuggingFace tokenizer
        max_tokens: Maximum tokens per chunk (default 510 for safety)
        overlap_sentences: Number of sentences to overlap between chunks
    
    Returns:
        List of text chunks with overlap
    """
    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    
    chunks = []
    i = 0
    
    while i < len(sentences):
        current_chunk = []
        current_token_count = 0
        
        # Add sentences until we hit the token limit
        for j in range(i, len(sentences)):
            sentence = sentences[j]
            sentence_tokens = tokenizer.encode(sentence, add_special_tokens=False)
            sentence_token_count = len(sentence_tokens)
            
            # Check if adding this sentence would exceed limit
            if current_token_count + sentence_token_count > max_tokens:
                break
            
            current_chunk.append(sentence)
            current_token_count += sentence_token_count
        
        # Save chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))
            
            # Move forward, but overlap by going back N sentences
            sentences_added = len(current_chunk)
            i += max(1, sentences_added - overlap_sentences)  # Ensure we always move forward
        else:
            # Single sentence exceeds limit - handle edge case
            if i < len(sentences):
                # Split long sentence by words
                long_sentence = sentences[i]
                word_chunks = _chunk_long_sentence(long_sentence, tokenizer, max_tokens)
                chunks.extend(word_chunks)
            i += 1
    
    return chunks


def _chunk_long_sentence(sentence, tokenizer, max_tokens):
    """Helper function to split a single very long sentence."""
    words = sentence.split()
    chunks = []
    current_chunk = []
    current_count = 0
    
    for word in words:
        word_tokens = len(tokenizer.encode(word, add_special_tokens=False))
        
        if current_count + word_tokens > max_tokens:
            if current_chunk:
                chunks.append(' '.join(current_chunk))
            current_chunk = [word]
            current_count = word_tokens
        else:
            current_chunk.append(word)
            current_count += word_tokens
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks


# Visualization helper
def visualize_overlap(chunks, max_preview=100):
    """Show how chunks overlap."""
    print(f"Total chunks: {len(chunks)}\n")
    
    for i, chunk in enumerate(chunks):
        print(f"{'='*80}")
        print(f"CHUNK {i+1}")
        print(f"{'='*80}")
        
        # Show beginning and end of chunk
        if len(chunk) > max_preview * 2:
            print(f"Start: {chunk[:max_preview]}...")
            print(f"End: ...{chunk[-max_preview:]}")
        else:
            print(chunk)
        
        # Show overlap with next chunk
        if i < len(chunks) - 1:
            next_chunk = chunks[i + 1]
            
            # Find common text (simplified - checks if end of current appears in start of next)
            overlap_found = False
            for length in range(min(len(chunk), len(next_chunk)), 0, -1):
                if chunk[-length:] == next_chunk[:length]:
                    print(f"\n🔗 OVERLAP WITH NEXT CHUNK ({length} chars):")
                    print(f"   '{chunk[-length:][:80]}...'")
                    overlap_found = True
                    break
            
            if not overlap_found:
                print("\nNo overlap detected with next chunk")
        
        print()

## Sketch code to run models on all tales. Final CSV outputs should have columns for tale ID, model name, emotion, score.
class MultiModelEmotionAnalyzer:
    """
    Analyze emotions across multiple transformer models for a collection of tales.
    """
    
    def __init__(self, max_tokens=510, overlap_sentences=2):
        """
        Initialize all three emotion models.
        
        Args:
            max_tokens: Maximum tokens per chunk (default 510 for safety)
            overlap_sentences: Number of sentences to overlap between chunks
        """
        print("Loading models...")
        
        # Use one tokenizer (all RoBERTa-based, so compatible)
        self.tokenizer = AutoTokenizer.from_pretrained(
            'j-hartmann/emotion-english-distilroberta-base'
        )
        
        # Load all three models
        self.models = {
            'distilroberta': pipeline(
                "text-classification", 
                model="j-hartmann/emotion-english-distilroberta-base",
                top_k=None,
                device=-1  # Use CPU; change to 0 for GPU
            ),
            'go_emotions': pipeline(
                "text-classification",
                model="SamLowe/roberta-base-go_emotions",
                top_k=None,
                device=-1
            ),
            'twitter_emotion': pipeline(
                "text-classification",
                model="cardiffnlp/twitter-roberta-base-emotion-latest",
                top_k=None,
                device=-1
            )
        }
        
        self.max_tokens = max_tokens
        self.overlap_sentences = overlap_sentences
        
        print("✓ All models loaded successfully")
    
    def chunk_text_with_overlap(self, text):
        """
        Chunk text respecting sentence boundaries with overlap.
        """
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        
        chunks = []
        i = 0
        
        while i < len(sentences):
            current_chunk = []
            current_token_count = 0
            
            # Add sentences until we hit the token limit
            for j in range(i, len(sentences)):
                sentence = sentences[j]
                sentence_tokens = self.tokenizer.encode(sentence, add_special_tokens=False)
                sentence_token_count = len(sentence_tokens)
                
                # Check if adding this sentence would exceed limit
                if current_token_count + sentence_token_count > self.max_tokens:
                    break
                
                current_chunk.append(sentence)
                current_token_count += sentence_token_count
            
            # Save chunk
            if current_chunk:
                chunks.append(' '.join(current_chunk))
                
                # Move forward with overlap
                sentences_added = len(current_chunk)
                i += max(1, sentences_added - self.overlap_sentences)
            else:
                # Handle edge case: single sentence too long
                if i < len(sentences):
                    chunks.append(sentences[i][:self.max_tokens * 4])  # Rough char estimate
                i += 1
        
        return chunks
    
    def analyze_single_tale(self, text, tale_title):
        """
        Analyze a single tale with all three models.
        
        Returns dict with results for each model.
        """
        chunks = self.chunk_text_with_overlap(text)
        
        results = {
            'tale': tale_title,
            'num_chunks': len(chunks),
            'total_chars': len(text),
            'chunk_details': []
        }
        
        # Analyze each chunk with all models
        for chunk_idx, chunk in enumerate(chunks):
            chunk_result = {
                'chunk_id': chunk_idx,
                'text_preview': chunk[:150] + '...' if len(chunk) > 150 else chunk,
                'token_count': len(self.tokenizer.encode(chunk))
            }
            
            # Run all three models on this chunk
            for model_name, model in self.models.items():
                emotions = model(chunk)
                chunk_result[model_name] = emotions[0]  # List of emotion predictions
            
            results['chunk_details'].append(chunk_result)
        
        return results
    
    def aggregate_tale_emotions(self, tale_results, model_name):
        """
        Aggregate emotion scores across all chunks for a single model.
        
        Returns sorted list of (emotion, avg_score) tuples.
        """
        emotion_scores = defaultdict(list)
        
        for chunk in tale_results['chunk_details']:
            for emotion_dict in chunk[model_name]:
                label = emotion_dict['label']
                score = emotion_dict['score']
                emotion_scores[label].append(score)
        
        # Calculate average score for each emotion
        avg_emotions = {
            emotion: sum(scores) / len(scores) 
            for emotion, scores in emotion_scores.items()
        }
        
        # Sort by average score
        return sorted(avg_emotions.items(), key=lambda x: x[1], reverse=True)
    
    def analyze_all_tales(self, tales_df):
        """
        Analyze all tales in the dataframe.
        
        Args:
            tales_df: DataFrame with columns 'book' (title) and 'text' (full text)
        
        Returns:
            DataFrame with emotion analysis results for all tales and all models
        """
        all_results = []
        
        print(f"\nAnalyzing {len(tales_df)} tales with 3 models...")
        
        # Use tqdm for progress bar
        for idx, row in tqdm(tales_df.iterrows(), total=len(tales_df)):
            tale_title = row['book']
            tale_text = row['text']
            
            # Analyze this tale
            tale_results = self.analyze_single_tale(tale_text, tale_title)
            
            # Aggregate emotions for each model
            result_row = {
                'tale': tale_title,
                'num_chunks': tale_results['num_chunks'],
                'total_chars': tale_results['total_chars']
            }
            
            # Add aggregated emotions for each model
            for model_name in self.models.keys():
                aggregated = self.aggregate_tale_emotions(tale_results, model_name)
                
                # Store top emotion and its score
                top_emotion, top_score = aggregated[0]
                result_row[f'{model_name}_top_emotion'] = top_emotion
                result_row[f'{model_name}_top_score'] = top_score
                
                # Store all emotion scores as dict
                result_row[f'{model_name}_all_emotions'] = dict(aggregated)
            
            all_results.append(result_row)
        
        return pd.DataFrame(all_results)
    
    def create_detailed_results(self, tales_df):
        """
        Create detailed results including chunk-level analysis.
        
        Returns:
            tuple: (summary_df, detailed_chunks_list)
        """
        all_tale_results = []
        all_chunk_results = []
        
        print(f"\nAnalyzing {len(tales_df)} tales with 3 models (detailed)...")
        
        for idx, row in tqdm(tales_df.iterrows(), total=len(tales_df)):
            tale_title = row['book']
            tale_text = row['text']
            
            # Analyze this tale
            tale_results = self.analyze_single_tale(tale_text, tale_title)
            
            # Summary for this tale
            tale_summary = {
                'tale': tale_title,
                'num_chunks': tale_results['num_chunks'],
                'total_chars': tale_results['total_chars']
            }
            
            # Aggregate and store for each model
            for model_name in self.models.keys():
                aggregated = self.aggregate_tale_emotions(tale_results, model_name)
                tale_summary[f'{model_name}_top_emotion'] = aggregated[0][0]
                tale_summary[f'{model_name}_top_score'] = aggregated[0][1]
                tale_summary[f'{model_name}_all_emotions'] = dict(aggregated)
            
            all_tale_results.append(tale_summary)
            
            # Store chunk-level details
            for chunk in tale_results['chunk_details']:
                chunk_row = {
                    'tale': tale_title,
                    'chunk_id': chunk['chunk_id'],
                    'text_preview': chunk['text_preview'],
                    'token_count': chunk['token_count']
                }
                
                # Add top emotion from each model for this chunk
                for model_name in self.models.keys():
                    emotions = chunk[model_name]
                    top = max(emotions, key=lambda x: x['score'])
                    chunk_row[f'{model_name}_emotion'] = top['label']
                    chunk_row[f'{model_name}_score'] = top['score']
                
                all_chunk_results.append(chunk_row)
        
        return pd.DataFrame(all_tale_results), pd.DataFrame(all_chunk_results)


# =============================================================================
# USAGE
# =============================================================================

# Initialize analyzer
analyzer = MultiModelEmotionAnalyzer(
    max_tokens=510,
    overlap_sentences=2
)

# Option 1: Simple summary (tale-level aggregated emotions)
results_df = analyzer.analyze_all_tales(tales)

print("\n" + "="*80)
print("RESULTS SUMMARY")
print("="*80)
print(results_df.head())

# Save results
results_df.to_csv('hca_emotion_analysis_summary.csv', index=False)
print("\n✓ Saved to 'hca_emotion_analysis_summary.csv'")


# Option 2: Detailed results (includes chunk-level analysis)
summary_df, chunks_df = analyzer.create_detailed_results(tales)

print("\n" + "="*80)
print("DETAILED RESULTS")
print("="*80)
print(f"Tales analyzed: {len(summary_df)}")
print(f"Total chunks: {len(chunks_df)}")

# Save detailed results
summary_df.to_csv('hca_emotion_summary.csv', index=False)
chunks_df.to_csv('hca_emotion_chunks.csv', index=False)
print("\n✓ Saved detailed results")


# =============================================================================
# ANALYSIS EXAMPLES
# =============================================================================

# Compare models: Which emotions do they prioritize?
print("\n" + "="*80)
print("MODEL COMPARISON: Top Emotions Across All Tales")
print("="*80)

for model in ['distilroberta', 'go_emotions', 'twitter_emotion']:
    top_emotions = results_df[f'{model}_top_emotion'].value_counts().head(5)
    print(f"\n{model.upper()}:")
    print(top_emotions)


# Find tales with highest emotion scores
print("\n" + "="*80)
print("MOST EMOTIONALLY INTENSE TALES (by model)")
print("="*80)

for model in ['distilroberta', 'go_emotions', 'twitter_emotion']:
    top_tale = results_df.nlargest(1, f'{model}_top_score')
    print(f"\n{model}: {top_tale['tale'].values[0]}")
    print(f"  Emotion: {top_tale[f'{model}_top_emotion'].values[0]}")
    print(f"  Score: {top_tale[f'{model}_top_score'].values[0]:.3f}")


# Explore specific tale
tale_name = "The Ugly Duckling"  # Change to any tale title
tale_data = results_df[results_df['tale'] == tale_name]

if not tale_data.empty:
    print(f"\n{'='*80}")
    print(f"EMOTION PROFILE: {tale_name}")
    print(f"{'='*80}")
    
    for model in ['distilroberta', 'go_emotions', 'twitter_emotion']:
        all_emotions = tale_data[f'{model}_all_emotions'].values[0]
        print(f"\n{model.upper()} - Top 5 emotions:")
        sorted_emotions = sorted(all_emotions.items(), key=lambda x: x[1], reverse=True)[:5]
        for emotion, score in sorted_emotions:
            print(f"  {emotion}: {score:.3f}")