"""
HCA Emotion Analysis - Laptop Version  
Quick test of transformer models on fairy tale text
"""

from transformers import pipeline
import torch

# Planned models:   distilroberta-base (https://huggingface.co/j-hartmann/emotion-english-distilroberta-base)
#                   roberta-base-go_emotions   
#                   cardiffnlp/twitter-roberta-base-sentiment-latest

models = [
    'j-hartmann/emotion-english-distilroberta-base',
    'cardiffnlp/twitter-roberta-base-sentiment-latest',
    'roberta-base-go_emotions'
]

# Outline test function
def test_emotion_analysis():
    print("Testing with distilroberta-base")
    
    try:
        # Load emotion model
        print("📥 Loading emotion model...")
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