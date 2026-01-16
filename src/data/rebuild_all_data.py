import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
import pickle

print("\n" + "="*80)
print("REBUILDING DATA - INCLUDING ALL 132 USERS")
print("="*80 + "\n")

print("STEP 1: LOADING RAW DATA")
print("-"*80)

users_df = pd.read_csv("data/raw/AudioShots_UAT.users.csv")
audio_df = pd.read_csv("data/raw/AudioShots_UAT.audios.csv")
likes_df = pd.read_csv("data/raw/AudioShots_UAT.likes.csv")
progress_df = pd.read_csv("data/raw/AudioShots_UAT.listeningprogresses.csv")

print(f"Loaded {len(users_df)} users (ALL USERS)")
print(f"Loaded {len(audio_df)} audios")
print(f"Loaded {len(likes_df)} likes")
print(f"Loaded {len(progress_df)} listening progress records\n")

print("STEP 2: BUILDING AUDIO METADATA")
print("-"*80)

audio_metadata = []

for idx, row in audio_df.iterrows():
    audio_id = row.get('_id')
    if pd.isna(audio_id):
        continue
    
    title = row.get('title', '')
    language = row.get('language.name', 'Unknown')
    
    # Get first non-null category
    category = None
    for i in range(5):
        cat = row.get(f'categories[{i}].name', None)
        if pd.notna(cat) and str(cat).strip():
            category = str(cat).strip()
            break
    
    if not category:
        category = 'General'
    
    audio_metadata.append({
        'audio_id': str(audio_id),
        'title': title,
        'language': language if pd.notna(language) else 'Unknown',
        'category': category
    })

audio_meta_df = pd.DataFrame(audio_metadata)

print(f"Created metadata for {len(audio_meta_df)} audios")
print(f"Unique languages: {audio_meta_df['language'].nunique()}")
print(f"Unique categories: {audio_meta_df['category'].nunique()}\n")

# Save audio metadata
Path("data/processed").mkdir(exist_ok=True)
audio_meta_df.to_csv("data/processed/audio_language_category.csv", index=False)
print("Saved: data/processed/audio_language_category.csv\n")

print("STEP 3: EXTRACTING LIKE INTERACTIONS")
print("-"*80)

user_col = None
audio_col = None

for col in likes_df.columns:
    if 'user' in col.lower() and 'id' in col.lower():
        user_col = col
    if 'audio' in col.lower() and 'id' in col.lower():
        audio_col = col

print(f"User column: {user_col}")
print(f"Audio column: {audio_col}")

interactions_likes = []

if user_col and audio_col:
    for idx, row in likes_df.iterrows():
        user_id = row.get(user_col)
        audio_id = row.get(audio_col)
        
        if pd.notna(user_id) and pd.notna(audio_id):
            interactions_likes.append({
                'user_id': str(user_id),
                'audio_id': str(audio_id),
                'score': 5.0
            })

print(f"Extracted {len(interactions_likes)} like interactions\n")

print("STEP 4: EXTRACTING LISTENING PROGRESS INTERACTIONS")
print("-"*80)

user_col = None
audio_col = None
progress_col = None
duration_col = None

for col in progress_df.columns:
    if 'user' in col.lower() and 'id' in col.lower():
        user_col = col
    if 'audio' in col.lower() and 'id' in col.lower():
        audio_col = col
    if col.lower() == 'progress':
        progress_col = col
    if col.lower() == 'duration':
        duration_col = col

print(f"User column: {user_col}")
print(f"Audio column: {audio_col}")
print(f"Progress column: {progress_col}")
print(f"Duration column: {duration_col}")

interactions_progress = []

if user_col and audio_col:
    for idx, row in progress_df.iterrows():
        user_id = row.get(user_col)
        audio_id = row.get(audio_col)
        progress = row.get(progress_col, 0) if progress_col else 0
        duration = row.get(duration_col, 0) if duration_col else 0
        
        if pd.notna(user_id) and pd.notna(audio_id):
            if duration > 0:
                completion = min(progress / duration, 1.0)
            else:
                completion = 0
            
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
            
            interactions_progress.append({
                'user_id': str(user_id),
                'audio_id': str(audio_id),
                'score': score
            })

print(f"Extracted {len(interactions_progress)} listening interactions\n")


print("STEP 5: LOADING PLAYLIST SIGNALS")
print("-"*80)

interactions_playlist = []

try:
    playlist_df = pd.read_csv("data/processed/playlist_signals.csv")
    if len(playlist_df) > 0:
        for idx, row in playlist_df.iterrows():
            user_id = row.get('user_id')
            audio_id = row.get('audio_id')
            score = row.get('playlist_add_score', 6.0)
            
            if pd.notna(user_id) and pd.notna(audio_id) and str(user_id) != 'unknown':
                interactions_playlist.append({
                    'user_id': str(user_id),
                    'audio_id': str(audio_id),
                    'score': float(score)
                })
        print(f"Loaded {len(interactions_playlist)} playlist interactions\n")
    else:
        print("Playlist signals file is empty\n")
except FileNotFoundError:
    print("No playlist signals found - continuing without them\n")
print("STEP 6: COMBINING ALL INTERACTIONS")
print("-"*80)

all_interactions = interactions_likes + interactions_progress + interactions_playlist

print(f"Interaction breakdown:")
print(f"  • From likes: {len(interactions_likes)}")
print(f"  • From progress: {len(interactions_progress)}")
print(f"  • From playlists: {len(interactions_playlist)}")
print(f"  • Total: {len(all_interactions)}\n")

if len(all_interactions) == 0:
    raise ValueError("ERROR: No interactions found! Cannot train model without user data.")

interactions_df = pd.DataFrame(all_interactions)

# Deduplicate - keep max score per user-audio pair
interactions_df = interactions_df.groupby(['user_id', 'audio_id'], as_index=False)['score'].max()

print(f"After deduplication: {len(interactions_df)} interactions")
print(f"Unique users with interactions: {interactions_df['user_id'].nunique()}")
print(f"Unique audios: {interactions_df['audio_id'].nunique()}\n")

print("STEP 7: MERGING WITH AUDIO METADATA")
print("-"*80)

interactions_df = interactions_df.merge(
    audio_meta_df[['audio_id', 'language', 'category']],
    on='audio_id',
    how='inner'
)

print(f"After merging: {len(interactions_df)} interactions")
print(f"Users: {interactions_df['user_id'].nunique()}")
print(f"Audios: {interactions_df['audio_id'].nunique()}\n")

print("STEP 8: FILTERING TRAINING USERS (>= 1 interaction)")
print("-"*80)

user_counts = interactions_df['user_id'].value_counts()
users_before = len(user_counts)

# CHANGED: Include users with 1+ interactions
valid_users = user_counts[user_counts >= 1].index
interactions_df = interactions_df[interactions_df['user_id'].isin(valid_users)]

users_after = interactions_df['user_id'].nunique()

print(f"Users with 0 interactions removed: {users_before - users_after}")
print(f"Users included in training: {users_after}")
print(f"Remaining interactions: {len(interactions_df)}\n")

# Save basic interactions
interactions_df.to_csv("data/processed/interactions.csv", index=False)
print("Saved: data/processed/interactions.csv\n")

print("STEP 9: ENCODING DATA - INCLUDING ALL USERS")
print("-"*80)

# Get ALL user IDs from raw users.csv
user_id_col = '_id'
all_user_ids = users_df[user_id_col].astype(str).tolist()

# CRITICAL FIX: Get unique user IDs from interactions too
users_in_interactions = interactions_df['user_id'].unique().tolist()

# Combine both sets - this ensures ALL users are included
all_unique_users = list(set(all_user_ids) | set(users_in_interactions))
all_unique_users.sort()  # Sort for consistency

print(f"Total users from raw data: {len(all_user_ids)}")
print(f"Users with interactions: {len(users_in_interactions)}")
print(f"Combined unique users: {len(all_unique_users)}\n")

# Debug: Check if interaction users are in raw data
users_only_in_interactions = set(users_in_interactions) - set(all_user_ids)
if users_only_in_interactions:
    print(f"  WARNING: {len(users_only_in_interactions)} users found in interactions but NOT in users.csv")
    print(f"  Sample: {list(users_only_in_interactions)[:3]}")
    print(f"  These will still be included in the encoder\n")

# Create user encoder with ALL users (from both sources)
user_encoder = LabelEncoder()
user_encoder.fit(all_unique_users)  # ← FIT ON ALL UNIQUE USERS

print(f"User encoder classes: {len(user_encoder.classes_)}")

# Now encode the interactions (only for users with data)
item_encoder = LabelEncoder()
language_encoder = LabelEncoder()
category_encoder = LabelEncoder()

interactions_df['user_idx'] = user_encoder.transform(interactions_df['user_id'])
interactions_df['item_idx'] = item_encoder.fit_transform(interactions_df['audio_id'])
interactions_df['language_idx'] = language_encoder.fit_transform(interactions_df['language'])
interactions_df['category_idx'] = category_encoder.fit_transform(interactions_df['category'])

print(f"Encoded {len(user_encoder.classes_)} users (ALL USERS from both sources)")
print(f"  - From users.csv: {len(all_user_ids)}")
print(f"  - From interactions: {len(users_in_interactions)}")
print(f"Encoded {interactions_df['item_idx'].nunique()} items")
print(f"Encoded {interactions_df['language_idx'].nunique()} languages")
print(f"Encoded {interactions_df['category_idx'].nunique()} categories\n")

print("STEP 10: CREATING ITEM METADATA LOOKUP")
print("-"*80)

item_metadata = interactions_df[['item_idx', 'audio_id', 'language_idx', 'category_idx']].drop_duplicates('item_idx')
item_metadata = item_metadata.sort_values('item_idx').reset_index(drop=True)

item_metadata.to_csv("data/processed/item_metadata.csv", index=False)
print(f"Saved: data/processed/item_metadata.csv ({len(item_metadata)} items)\n")

print("STEP 11: SAVING ML INTERACTIONS")
print("-"*80)

ml_interactions = interactions_df[['user_idx', 'item_idx', 'language_idx', 'category_idx', 'score']]
ml_interactions.to_csv("data/processed/ml_interactions.csv", index=False)

print(f"Saved: data/processed/ml_interactions.csv\n")

# ============================================================================
# STEP 12: Save Encoders
# ============================================================================
print("STEP 12: SAVING ENCODERS")
print("-"*80)

Path("models_saved").mkdir(exist_ok=True)

with open("models_saved/user_encoder.pkl", "wb") as f:
    pickle.dump(user_encoder, f)
print("Saved: models_saved/user_encoder.pkl")

with open("models_saved/item_encoder.pkl", "wb") as f:
    pickle.dump(item_encoder, f)
print("Saved: models_saved/item_encoder.pkl")

with open("models_saved/language_encoder.pkl", "wb") as f:
    pickle.dump(language_encoder, f)
print("Saved: models_saved/language_encoder.pkl")

with open("models_saved/category_encoder.pkl", "wb") as f:
    pickle.dump(category_encoder, f)
print("Saved: models_saved/category_encoder.pkl\n")

print("STEP 13: CREATING USER PREFERENCE MAPS")
print("-"*80)

# User-Language Map
user_lang_map = interactions_df.groupby('user_id')['language'].agg(
    lambda x: x.value_counts().index[0]
).reset_index()
user_lang_map.columns = ['user_id', 'language_id']
user_lang_map.to_csv("data/processed/user_language_map.csv", index=False)
print("Saved: data/processed/user_language_map.csv")

# User-Category Map
user_cat_map = interactions_df.groupby('user_id')['category'].agg(
    lambda x: x.value_counts().index[0]
).reset_index()
user_cat_map.columns = ['user_id', 'category']
user_cat_map.to_csv("data/processed/user_category_map.csv", index=False)
print("Saved: data/processed/user_category_map.csv")

# User-Category Index
user_cat_idx = interactions_df.groupby('user_idx')['category_idx'].agg(
    lambda x: x.value_counts().index[0]
).reset_index()
user_cat_idx.columns = ['user_idx', 'category_idx']
user_cat_idx.to_csv("data/processed/user_category_idx.csv", index=False)
print("Saved: data/processed/user_category_idx.csv\n")

print("STEP 14: CREATING COLD START USER LIST")
print("-"*80)

users_with_interactions = set(interactions_df['user_id'].unique())
all_encoded_users = set(user_encoder.classes_)
cold_start_users = list(all_encoded_users - users_with_interactions)

cold_start_df = pd.DataFrame({
    'user_id': cold_start_users,
    'has_interactions': False
})

cold_start_df.to_csv("data/processed/cold_start_users.csv", index=False)
print(f"Saved: data/processed/cold_start_users.csv ({len(cold_start_users)} users)\n")


print("FINAL STATISTICS:")
print(f"  • Total users in system: {len(user_encoder.classes_)}")
print(f"  • Users with interactions (for training): {interactions_df['user_idx'].nunique()}")
print(f"  • Cold start users (no interactions): {len(cold_start_users)}")
print(f"  • Items: {interactions_df['item_idx'].nunique()}")
print(f"  • Languages: {interactions_df['language_idx'].nunique()}")
print(f"  • Categories: {interactions_df['category_idx'].nunique()}")
print(f"  • Total interactions: {len(interactions_df)}")
print()

print("SCORE DISTRIBUTION:")
print(interactions_df['score'].value_counts().sort_index())
print()

print("FILES CREATED:")
print("  data/processed/audio_language_category.csv")
print("  data/processed/interactions.csv")
print("  data/processed/ml_interactions.csv")
print("  data/processed/item_metadata.csv")
print("  data/processed/user_language_map.csv")
print("  data/processed/user_category_map.csv")
print("  data/processed/user_category_idx.csv")
print("  data/processed/cold_start_users.csv")
print("  models_saved/user_encoder.pkl")
print("  models_saved/item_encoder.pkl")
print("  models_saved/language_encoder.pkl")
print("  models_saved/category_encoder.pkl")
print()
print("You can now retrain the model with the updated data including all users.")