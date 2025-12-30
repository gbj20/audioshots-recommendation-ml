"""
Complete Rebuild Pipeline - Include ALL Audio IDs
Run this to rebuild your training data with all 9,983 audio IDs
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pickle

print("\n" + "="*80)
print("REBUILDING TRAINING DATA - COMPLETE PIPELINE")
print("="*80 + "\n")

# ---------------- STEP 1: Load All Raw Data ----------------
print("STEP 1: LOADING RAW DATA")
print("-"*80)

audio_df = pd.read_csv("data/raw/AudioShots_UAT.audios.csv")
likes_df = pd.read_csv("data/raw/AudioShots_UAT.likes.csv")
progress_df = pd.read_csv("data/raw/AudioShots_UAT.listeningprogresses.csv")
users_df = pd.read_csv("data/raw/AudioShots_UAT.users.csv")

print(f"✓ Loaded {len(audio_df)} audios")
print(f"✓ Loaded {len(likes_df)} likes")
print(f"✓ Loaded {len(progress_df)} listening progress records")
print(f"✓ Loaded {len(users_df)} users")

# Debug: Show column names
print(f"\n📋 Likes columns: {list(likes_df.columns)}")
print(f"📋 Progress columns: {list(progress_df.columns)}\n")

# ---------------- STEP 2: Build Audio-Language-Category Mapping ----------------
print("STEP 2: BUILDING AUDIO-LANGUAGE-CATEGORY MAPPING")
print("-"*80)

audio_lang_cat_rows = []

for idx, row in audio_df.iterrows():
    audio_id = row.get('_id')
    title = row.get('title', '')
    language = row.get('language.name', None)
    
    if pd.isna(audio_id):
        continue
    
    # Get all categories (up to 5)
    categories = []
    for i in range(5):
        cat = row.get(f'categories[{i}].name', None)
        if pd.notna(cat) and str(cat).strip():
            categories.append(str(cat).strip())
    
    # Create one row per category
    if categories:
        for category in categories:
            audio_lang_cat_rows.append({
                'audio_id': audio_id,
                'title': title,
                'language': language if pd.notna(language) else 'Unknown',
                'category': category
            })
    else:
        # Even if no category, include the audio
        audio_lang_cat_rows.append({
            'audio_id': audio_id,
            'title': title,
            'language': language if pd.notna(language) else 'Unknown',
            'category': 'General'
        })

audio_lc_df = pd.DataFrame(audio_lang_cat_rows)
audio_lc_df.to_csv("data/processed/audio_language_category.csv", index=False)

print(f"✓ Created {len(audio_lc_df)} audio-language-category mappings")
print(f"✓ Unique audio IDs: {audio_lc_df['audio_id'].nunique()}")
print(f"✓ Languages: {audio_lc_df['language'].nunique()}")
print(f"✓ Categories: {audio_lc_df['category'].nunique()}")
print(f"✓ Saved to: data/processed/audio_language_category.csv\n")

# ---------------- STEP 3: Build Interactions from Likes ----------------
print("STEP 3: BUILDING INTERACTIONS FROM LIKES")
print("-"*80)

# Find correct column names for user_id and audio_id in likes
user_col_likes = None
audio_col_likes = None

for col in likes_df.columns:
    if 'user' in col.lower() and 'id' in col.lower():
        user_col_likes = col
    if 'audio' in col.lower() and 'id' in col.lower():
        audio_col_likes = col

print(f"📋 User column in likes: {user_col_likes}")
print(f"📋 Audio column in likes: {audio_col_likes}")

interactions_from_likes = []

if user_col_likes and audio_col_likes:
    for idx, row in likes_df.iterrows():
        user_id = row.get(user_col_likes)
        audio_id = row.get(audio_col_likes)
        
        if pd.notna(user_id) and pd.notna(audio_id):
            interactions_from_likes.append({
                'user_id': str(user_id),
                'audio_id': str(audio_id),
                'score': 5.0
            })
    print(f"✓ Created {len(interactions_from_likes)} interactions from likes\n")
else:
    print(f"⚠ Could not find user/audio columns in likes\n")

# ---------------- STEP 4: Build Interactions from Listening Progress ----------------
print("STEP 4: BUILDING INTERACTIONS FROM LISTENING PROGRESS")
print("-"*80)

# Find correct column names
user_col_progress = None
audio_col_progress = None
progress_col = None
duration_col = None

for col in progress_df.columns:
    if 'user' in col.lower() and 'id' in col.lower():
        user_col_progress = col
    if 'audio' in col.lower() and 'id' in col.lower():
        audio_col_progress = col
    if col.lower() == 'progress':
        progress_col = col
    if col.lower() == 'duration':
        duration_col = col

print(f"📋 User column in progress: {user_col_progress}")
print(f"📋 Audio column in progress: {audio_col_progress}")
print(f"📋 Progress column: {progress_col}")
print(f"📋 Duration column: {duration_col}")

interactions_from_progress = []

if user_col_progress and audio_col_progress:
    for idx, row in progress_df.iterrows():
        user_id = row.get(user_col_progress)
        audio_id = row.get(audio_col_progress)
        progress = row.get(progress_col, 0) if progress_col else 0
        duration = row.get(duration_col, 0) if duration_col else 0
        
        if pd.notna(user_id) and pd.notna(audio_id):
            # Calculate completion percentage
            if duration > 0:
                completion = min(progress / duration, 1.0)
            else:
                completion = 0
            
            # Score based on completion
            if completion >= 0.8:
                score = 5.0
            elif completion >= 0.6:
                score = 4.0
            elif completion >= 0.4:
                score = 3.0
            elif completion >= 0.2:
                score = 2.0
            else:
                score = 1.0
            
            interactions_from_progress.append({
                'user_id': str(user_id),
                'audio_id': str(audio_id),
                'score': score
            })
    print(f"✓ Created {len(interactions_from_progress)} interactions from listening progress\n")
else:
    print(f"⚠ Could not find user/audio columns in progress\n")

# ---------------- STEP 5: Combine Interactions ----------------
print("STEP 5: COMBINING INTERACTIONS")
print("-"*80)

# Combine all interactions
all_interactions = interactions_from_likes + interactions_from_progress

if len(all_interactions) == 0:
    print("⚠ WARNING: No real interactions found!")
    print("⚠ Will create synthetic interactions for all audios\n")
    
    # Create a minimal interactions dataframe
    interactions_df = pd.DataFrame(columns=['user_id', 'audio_id', 'score'])
else:
    interactions_df = pd.DataFrame(all_interactions)
    
    # Deduplicate: if same user-audio pair, keep max score
    interactions_df = interactions_df.groupby(['user_id', 'audio_id'], as_index=False)['score'].max()
    
    print(f"✓ Total interactions after deduplication: {len(interactions_df)}")
    print(f"✓ Unique users: {interactions_df['user_id'].nunique()}")
    print(f"✓ Unique audios: {interactions_df['audio_id'].nunique()}\n")

# Save basic interactions
interactions_df.to_csv("data/processed/interactions.csv", index=False)
print(f"✓ Saved to: data/processed/interactions.csv\n")

# ---------------- STEP 6: Create Synthetic Interactions for All Audios ----------------
print("STEP 6: CREATING SYNTHETIC INTERACTIONS FOR ALL AUDIOS")
print("-"*80)

# Get all audio IDs
all_audio_ids = audio_lc_df['audio_id'].unique()

# Get or create users
if len(interactions_df) > 0 and 'user_id' in interactions_df.columns:
    # Use existing users
    available_users = interactions_df['user_id'].unique()
    print(f"✓ Using {len(available_users)} existing users from interactions")
else:
    # Create synthetic users from users table
    if len(users_df) > 0:
        # Look for _id or userId column
        user_id_col = None
        for col in users_df.columns:
            if col == '_id' or 'userId' in col or 'user_id' in col:
                user_id_col = col
                break
        
        if user_id_col:
            available_users = users_df[user_id_col].dropna().unique()
            print(f"✓ Using {len(available_users)} users from users table")
        else:
            # Create generic users
            available_users = [f"synthetic_user_{i}" for i in range(min(10, len(users_df)))]
            print(f"✓ Created {len(available_users)} synthetic users")
    else:
        # Create generic users
        available_users = [f"synthetic_user_{i}" for i in range(10)]
        print(f"✓ Created {len(available_users)} synthetic users")

# Create one interaction per audio with language and category
print(f"\n⚠ Creating synthetic interactions for {len(all_audio_ids)} audios...")

synthetic_rows = []
for audio_id in all_audio_ids:
    # Get language and category for this audio (take first row if multiple)
    audio_info = audio_lc_df[audio_lc_df['audio_id'] == audio_id].iloc[0]
    
    # Assign to a random user
    user = np.random.choice(available_users)
    
    synthetic_rows.append({
        'user_id': str(user),
        'audio_id': str(audio_id),
        'score': 3.0,  # Neutral score
        'language': audio_info['language'],
        'category': audio_info['category']
    })

synthetic_df = pd.DataFrame(synthetic_rows)

# Combine with real interactions if any
if len(interactions_df) > 0:
    # Merge with language-category info
    interactions_enriched = interactions_df.merge(
        audio_lc_df[['audio_id', 'language', 'category']].drop_duplicates('audio_id'),
        on='audio_id',
        how='left'
    )
    
    # Combine with synthetic
    enriched = pd.concat([interactions_enriched, synthetic_df], ignore_index=True)
    
    # Deduplicate again
    enriched = enriched.groupby(['user_id', 'audio_id'], as_index=False).agg({
        'score': 'max',
        'language': 'first',
        'category': 'first'
    })
else:
    enriched = synthetic_df

# Fill any missing language/category
enriched['language'] = enriched['language'].fillna('Unknown')
enriched['category'] = enriched['category'].fillna('General')

print(f"\n✓ Total enriched interactions: {len(enriched)}")
print(f"✓ Covering {enriched['audio_id'].nunique()} unique audios")
print(f"✓ Using {enriched['user_id'].nunique()} unique users\n")

# ---------------- STEP 7: Encode Everything ----------------
print("STEP 7: ENCODING ALL DATA")
print("-"*80)

# Create encoders
user_encoder = LabelEncoder()
item_encoder = LabelEncoder()
language_encoder = LabelEncoder()
category_encoder = LabelEncoder()

# Fit encoders
enriched['user_idx'] = user_encoder.fit_transform(enriched['user_id'])
enriched['item_idx'] = item_encoder.fit_transform(enriched['audio_id'])
enriched['language_idx'] = language_encoder.fit_transform(enriched['language'])
enriched['category_idx'] = category_encoder.fit_transform(enriched['category'])

print(f"✓ Encoded {enriched['user_idx'].nunique()} users")
print(f"✓ Encoded {enriched['item_idx'].nunique()} items (audio IDs)")
print(f"✓ Encoded {enriched['language_idx'].nunique()} languages")
print(f"✓ Encoded {enriched['category_idx'].nunique()} categories\n")

# Save ML interactions
ml_interactions = enriched[['user_idx', 'item_idx', 'language_idx', 'category_idx', 'score']]
ml_interactions.to_csv("data/processed/ml_interactions.csv", index=False)
print(f"✓ Saved to: data/processed/ml_interactions.csv\n")

# ---------------- STEP 8: Save Encoders ----------------
print("STEP 8: SAVING ENCODERS")
print("-"*80)

Path("models_saved").mkdir(exist_ok=True)

with open("models_saved/user_encoder.pkl", "wb") as f:
    pickle.dump(user_encoder, f)
print("✓ Saved user_encoder.pkl")

with open("models_saved/item_encoder.pkl", "wb") as f:
    pickle.dump(item_encoder, f)
print("✓ Saved item_encoder.pkl")

with open("models_saved/language_encoder.pkl", "wb") as f:
    pickle.dump(language_encoder, f)
print("✓ Saved language_encoder.pkl")

with open("models_saved/category_encoder.pkl", "wb") as f:
    pickle.dump(category_encoder, f)
print("✓ Saved category_encoder.pkl\n")

# ---------------- STEP 9: Build User Maps ----------------
print("STEP 9: BUILDING USER PREFERENCE MAPS")
print("-"*80)

# User-Language Map
user_lang_map = enriched.groupby('user_id')['language'].agg(
    lambda x: x.value_counts().index[0]  # Most common language
).reset_index()
user_lang_map.columns = ['user_id', 'language_id']
user_lang_map.to_csv("data/processed/user_language_map.csv", index=False)
print("✓ Saved user_language_map.csv")

# User-Category Map
user_cat_map = enriched.groupby('user_id')['category'].agg(
    lambda x: x.value_counts().index[0]  # Most common category
).reset_index()
user_cat_map.columns = ['user_id', 'category']
user_cat_map.to_csv("data/processed/user_category_map.csv", index=False)
print("✓ Saved user_category_map.csv")

# User-Category Index
user_cat_idx = enriched.groupby('user_idx')['category_idx'].agg(
    lambda x: x.value_counts().index[0]  # Most common category index
).reset_index()
user_cat_idx.columns = ['user_idx', 'category_idx']
user_cat_idx.to_csv("data/processed/user_category_idx.csv", index=False)
print("✓ Saved user_category_idx.csv\n")

# ---------------- STEP 10: Summary ----------------
print("="*80)
print("REBUILD COMPLETE!")
print("="*80 + "\n")

print("📊 FINAL STATISTICS:")
print(f"  • Total users: {enriched['user_idx'].nunique()}")
print(f"  • Total items (audios): {enriched['item_idx'].nunique()}")
print(f"  • Total languages: {enriched['language_idx'].nunique()}")
print(f"  • Total categories: {enriched['category_idx'].nunique()}")
print(f"  • Total interactions: {len(enriched)}")
print(f"  • Real interactions: {len(interactions_df) if len(interactions_df) > 0 else 0}")
print(f"  • Synthetic interactions: {len(synthetic_df)}")
print()

print("✅ FILES CREATED:")
print("  • data/processed/audio_language_category.csv")
print("  • data/processed/interactions.csv")
print("  • data/processed/ml_interactions.csv")
print("  • data/processed/user_language_map.csv")
print("  • data/processed/user_category_map.csv")
print("  • data/processed/user_category_idx.csv")
print("  • models_saved/user_encoder.pkl")
print("  • models_saved/item_encoder.pkl")
print("  • models_saved/language_encoder.pkl")
print("  • models_saved/category_encoder.pkl")
print()

print("🎯 NEXT STEPS:")
print("  1. Run: python src/training/train_model.py")
print("  2. Run: python src/test/test_v3_language_category.py")
print()
print("="*80 + "\n")