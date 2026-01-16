import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

import pandas as pd
import pickle
from src.inference.recommend import recommend_with_details

USER_ENCODER_PATH = "models_saved/user_encoder.pkl"
DATA_PATH = "data/processed/ml_interactions.csv"


def visualize_user_recommendations(user_id, num_recommendations=10):
    """Display recommendations for a single user with formatting"""
    
    print("\n" + "="*100)
    print(f"RECOMMENDATIONS FOR USER: {user_id}")
    print("="*100)
    
    # Get user's interaction history
    df = pd.read_csv(DATA_PATH)
    
    with open(USER_ENCODER_PATH, "rb") as f:
        user_encoder = pickle.load(f)
    
    if user_id in user_encoder.classes_:
        user_idx = user_encoder.transform([user_id])[0]
        user_history = df[df['user_idx'] == user_idx]
        
        print(f"\nUser History:")
        print(f"  Total interactions: {len(user_history)}")
        print(f"  Average score: {user_history['score'].mean():.2f}")
        print(f"  Score distribution:")
        for score, count in user_history['score'].value_counts().sort_index().items():
            print(f"    Score {score}: {count} items")
    
    # Get recommendations
    print(f"\n{'='*100}")
    print(f"TOP {num_recommendations} RECOMMENDATIONS")
    print(f"{'='*100}\n")
    
    recommendations = recommend_with_details(user_id, top_k=num_recommendations)
    
    if len(recommendations) == 0:
        print("  ⚠ No recommendations available for this user\n")
        return
    
    # Display recommendations in a nice format
    for i, row in recommendations.iterrows():
        print(f"#{i+1:2d} │ {row['title'][:60]:<60}")
        print(f"    │ ID: {row['audio_id']:<25} │ Score: {row['predicted_score']:<8}")
        print(f"    │ Language: {row['language']:<20} │ Category: {row['category']:<30}")
        print(f"    {'─'*98}")
    
    print()


def compare_recommendations_across_users(num_users=5, num_recs=5):
    """Compare recommendations across multiple users"""
    
    print("\n" + "="*100)
    print("COMPARING RECOMMENDATIONS ACROSS USERS")
    print("="*100 + "\n")
    
    with open(USER_ENCODER_PATH, "rb") as f:
        user_encoder = pickle.load(f)
    
    # Get sample users
    num_users = min(num_users, len(user_encoder.classes_))
    sample_user_indices = range(num_users)
    sample_users = user_encoder.inverse_transform(sample_user_indices)
    
    for user_id in sample_users:
        print(f"\nUser: {user_id}")
        print("-"*100)
        
        recs = recommend_with_details(user_id, top_k=num_recs)
        
        if len(recs) > 0:
            for i, row in recs.iterrows():
                print(f"  {i+1}. {row['title'][:50]:<50} | {row['language']:<12} | {row['category']:<20}")
        else:
            print("  No recommendations")
    
    print("\n" + "="*100 + "\n")


def analyze_recommendation_diversity():
    """Analyze diversity of recommendations across all users"""
    
    print("\n" + "="*100)
    print("ANALYZING RECOMMENDATION DIVERSITY")
    print("="*100 + "\n")
    
    with open(USER_ENCODER_PATH, "rb") as f:
        user_encoder = pickle.load(f)
    
    all_recommended_items = set()
    language_distribution = {}
    category_distribution = {}
    
    num_users_to_test = min(20, len(user_encoder.classes_))
    
    print(f"Testing recommendations for {num_users_to_test} users...\n")
    
    for i in range(num_users_to_test):
        user_id = user_encoder.inverse_transform([i])[0]
        recs = recommend_with_details(user_id, top_k=10)
        
        if len(recs) > 0:
            all_recommended_items.update(recs['audio_id'].values)
            
            for lang in recs['language'].values:
                language_distribution[lang] = language_distribution.get(lang, 0) + 1
            
            for cat in recs['category'].values:
                category_distribution[cat] = category_distribution.get(cat, 0) + 1
    
    print(f"DIVERSITY METRICS:")
    print(f"  Unique items recommended: {len(all_recommended_items)}")
    print()
    
    print(f"LANGUAGE DISTRIBUTION IN RECOMMENDATIONS:")
    for lang, count in sorted(language_distribution.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {lang:<20}: {count:>4} times")
    print()
    
    print(f"CATEGORY DISTRIBUTION IN RECOMMENDATIONS:")
    for cat, count in sorted(category_distribution.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {cat:<30}: {count:>4} times")
    print()
    
    print("="*100 + "\n")


if __name__ == "__main__":
    print("\n" + "="*100)
    print("RECOMMENDATION VISUALIZATION TOOL")
    print("="*100)
    
    # Load user encoder to get sample users
    with open(USER_ENCODER_PATH, "rb") as f:
        user_encoder = pickle.load(f)
    
    # Show menu
    print("\nOptions:")
    print("  1. View detailed recommendations for specific users")
    print("  2. Compare recommendations across multiple users")
    print("  3. Analyze recommendation diversity")
    print("  4. All of the above")
    
    choice = input("\nEnter choice (1-4) [default=4]: ").strip() or "4"
    
    if choice == "1":
        # Get first user as example
        user_id = user_encoder.inverse_transform([0])[0]
        visualize_user_recommendations(user_id, num_recommendations=10)
    
    elif choice == "2":
        compare_recommendations_across_users(num_users=5, num_recs=5)
    
    elif choice == "3":
        analyze_recommendation_diversity()
    
    else:  # choice == "4" or anything else
        # Show everything
        user_id = user_encoder.inverse_transform([0])[0]
        visualize_user_recommendations(user_id, num_recommendations=10)
        compare_recommendations_across_users(num_users=5, num_recs=5)
        analyze_recommendation_diversity()
    
    print("="*100)
    print("Visualization complete!")
    print("="*100 + "\n")