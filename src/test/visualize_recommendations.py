"""
Visualization Tool for Audio Recommendations
This script shows the language and category distribution of recommended audios
"""

import sys
from pathlib import Path

# Add project root to path so we can import our modules
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

import pandas as pd
import pickle
from collections import Counter
import matplotlib.pyplot as plt
from src.inference.recommend import recommend_for_user


AUDIO_DATA_PATH = "data/processed/audio_language_category.csv"
ITEM_ENCODER_PATH = "models_saved/item_encoder.pkl"
USER_ENCODER_PATH = "models_saved/user_encoder.pkl"

def load_data():
    """Load audio metadata and encoders"""
    print("\n" + "="*70)
    print("LOADING DATA...")
    print("="*70 + "\n")
    
    # Load audio metadata (has language and category for each audio)
    audio_df = pd.read_csv(AUDIO_DATA_PATH)
    print(f"✓ Loaded {len(audio_df)} audio metadata records")
    
    # Load item encoder (converts audio_id to index and vice versa)
    with open(ITEM_ENCODER_PATH, "rb") as f:
        item_encoder = pickle.load(f)
    print(f"✓ Loaded item encoder with {len(item_encoder.classes_)} items")
    
    # Load user encoder (to get list of users)
    with open(USER_ENCODER_PATH, "rb") as f:
        user_encoder = pickle.load(f)
    print(f"✓ Loaded user encoder with {len(user_encoder.classes_)} users\n")
    
    return audio_df, item_encoder, user_encoder


# ============================================
# STEP 2: GET RECOMMENDATIONS FOR A USER
# ============================================
def get_recommendations_with_metadata(user_id, audio_df, top_k=10):
    """
    Get recommendations for a user and add language/category info
    
    Args:
        user_id: The user to recommend for
        audio_df: DataFrame with audio metadata
        top_k: Number of recommendations
        
    Returns:
        DataFrame with columns: audio_id, language, category
    """
    print(f"Getting recommendations for user: {user_id}")
    
    # Get recommended audio IDs from the model
    recommended_audio_ids = recommend_for_user(user_id, top_k=top_k)
    print(f"✓ Got {len(recommended_audio_ids)} recommendations\n")
    
    # Create a list to store results
    results = []
    
    # For each recommended audio, find its language and category
    for audio_id in recommended_audio_ids:
        # Find this audio in the metadata
        audio_info = audio_df[audio_df['audio_id'] == audio_id]
        
        if not audio_info.empty:
            # Get the first row (in case audio has multiple categories)
            info = audio_info.iloc[0]
            results.append({
                'audio_id': audio_id,
                'language': info['language'],
                'category': info['category']
            })
        else:
            # If not found, mark as unknown
            results.append({
                'audio_id': audio_id,
                'language': 'Unknown',
                'category': 'Unknown'
            })
    
    return pd.DataFrame(results)


def visualize_recommendations(recommendations_df, user_id):
    """
    Create visualizations showing language and category distribution
    
    Args:
        recommendations_df: DataFrame with audio_id, language, category
        user_id: The user ID (for the title)
    """
    print("="*70)
    print("VISUALIZATION")
    print("="*70 + "\n")
    
    # Count languages and categories
    language_counts = Counter(recommendations_df['language'])
    category_counts = Counter(recommendations_df['category'])
    
    # Create figure with 2 subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f'Recommendation Analysis for User: {user_id}', fontsize=16, fontweight='bold')
    
    # === LEFT PLOT: Language Distribution ===
    languages = list(language_counts.keys())
    lang_values = list(language_counts.values())
    colors1 = plt.cm.Set3(range(len(languages)))
    
    ax1.bar(languages, lang_values, color=colors1, edgecolor='black')
    ax1.set_xlabel('Language', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Number of Recommendations', fontsize=12, fontweight='bold')
    ax1.set_title('Distribution by Language', fontsize=14)
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, v in enumerate(lang_values):
        ax1.text(i, v + 0.1, str(v), ha='center', va='bottom', fontweight='bold')
    
    # === RIGHT PLOT: Category Distribution ===
    categories = list(category_counts.keys())
    cat_values = list(category_counts.values())
    colors2 = plt.cm.Pastel1(range(len(categories)))
    
    ax2.bar(categories, cat_values, color=colors2, edgecolor='black')
    ax2.set_xlabel('Category', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Number of Recommendations', fontsize=12, fontweight='bold')
    ax2.set_title('Distribution by Category', fontsize=14)
    ax2.grid(axis='y', alpha=0.3)
    
    # Rotate category labels if too many
    if len(categories) > 3:
        ax2.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for i, v in enumerate(cat_values):
        ax2.text(i, v + 0.1, str(v), ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    # Save the figure
    output_path = f"visualization_user_{user_id}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Visualization saved as: {output_path}")
    
    # Show the plot
    plt.show()


# ============================================
# STEP 4: PRINT DETAILED TABLE
# ============================================
def print_detailed_table(recommendations_df):
    """Print a nice table showing all recommendations"""
    print("\n" + "="*70)
    print("DETAILED RECOMMENDATIONS")
  
    
    print(f"{'No.':<5} {'Audio ID':<30} {'Language':<15} {'Category':<20}")
    print("-"*70)
    
    for idx, row in recommendations_df.iterrows():
        print(f"{idx+1:<5} {row['audio_id']:<30} {row['language']:<15} {row['category']:<20}")
    
    print("\n" + "="*70)


# ============================================
# MAIN FUNCTION
# ============================================
def main():
    """Main function to run the visualization"""
    
    print("\n" + "="*70)
    print("AUDIO RECOMMENDATION VISUALIZER")
    print("="*70)
    
    # Load data
    audio_df, item_encoder, user_encoder = load_data()
    
    # Get list of available users
    available_users = user_encoder.classes_
    
    print("="*70)
    print("AVAILABLE USERS")
    print("="*70 + "\n")
    print(f"Total users in system: {len(available_users)}")
    print(f"First 5 user IDs: {list(available_users[:5])}\n")
    
    # Let user choose which user to analyze
    print("Enter a user ID to analyze (or press Enter to use the first user):")
    user_input = input("> ").strip()
    
    if user_input == "":
        user_id = available_users[0]
        print(f"Using default user: {user_id}\n")
    elif user_input in available_users:
        user_id = user_input
    else:
        print(f"⚠ User '{user_input}' not found. Using default user: {available_users[0]}\n")
        user_id = available_users[0]
    
    # Get recommendations with metadata
    recommendations_df = get_recommendations_with_metadata(user_id, audio_df, top_k=10)
    
    # Print detailed table
    print_detailed_table(recommendations_df)
    
    # Print summary statistics
    print("\nSUMMARY STATISTICS")
    print("-"*70)
    print(f"Total recommendations: {len(recommendations_df)}")
    print(f"Unique languages: {recommendations_df['language'].nunique()}")
    print(f"Unique categories: {recommendations_df['category'].nunique()}")
    
    print("\nLanguage breakdown:")
    for lang, count in recommendations_df['language'].value_counts().items():
        print(f"  • {lang}: {count}")
    
    print("\nCategory breakdown:")
    for cat, count in recommendations_df['category'].value_counts().items():
        print(f"  • {cat}: {count}")
    print()
    
    # Create visualization
    visualize_recommendations(recommendations_df, user_id)
    
    print("\n✓ Analysis complete!\n")


if __name__ == "__main__":
    main()