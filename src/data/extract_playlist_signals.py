"""
Extract Playlist Add Signals
User adding audio to playlist = strong intent signal
"""

import pandas as pd
import numpy as np
from pathlib import Path

print("="*70)
print("EXTRACTING PLAYLIST ADD SIGNALS")
print("="*70 + "\n")

# Load data
try:
    playlist_df = pd.read_csv("data/raw/Audioshots_Production.savedplaylistaudios.csv")
    print(f"Loaded {len(playlist_df)} playlist entries")
    print(f"Columns: {list(playlist_df.columns)}\n")
    
    # Find column names
    user_col = None
    audio_col = None
    playlist_col = None
    
    for col in playlist_df.columns:
        if 'user' in col.lower() and 'id' in col.lower():
            user_col = col
        if 'audio' in col.lower() and 'id' in col.lower():
            audio_col = col
        if 'playlist' in col.lower() and 'id' in col.lower():
            playlist_col = col
    
    print(f"User column: {user_col}")
    print(f"Audio column: {audio_col}")
    print(f"Playlist column: {playlist_col}\n")
    
    if not audio_col:
        print("ERROR: Cannot find audio column")
        exit(1)
    
    # Extract playlist adds
    playlist_signals = []
    
    for idx, row in playlist_df.iterrows():
        user_id = row.get(user_col) if user_col else None
        audio_id = row.get(audio_col)
        playlist_id = row.get(playlist_col) if playlist_col else None
        
        if pd.notna(audio_id):
            playlist_signals.append({
                'user_id': str(user_id) if pd.notna(user_id) else 'unknown',
                'audio_id': str(audio_id),
                'playlist_id': str(playlist_id) if pd.notna(playlist_id) else 'unknown',
                'playlist_add_score': 6.0  # Strong positive signal
            })
    
    playlist_df_final = pd.DataFrame(playlist_signals)
    
    # Deduplicate (same user adding same audio to multiple playlists)
    # Keep max score per user-audio pair
    playlist_df_final = playlist_df_final.groupby(['user_id', 'audio_id'], as_index=False).agg({
        'playlist_add_score': 'max'
    })
    
    # Save
    Path("data/processed").mkdir(exist_ok=True)
    playlist_df_final.to_csv("data/processed/playlist_signals.csv", index=False)
    
    print(f"RESULTS:")
    print(f"  Total playlist adds: {len(playlist_df_final)}")
    print(f"  Unique users: {playlist_df_final['user_id'].nunique()}")
    print(f"  Unique audios: {playlist_df_final['audio_id'].nunique()}")
    
    print("\nSaved to: data/processed/playlist_signals.csv")
    
except FileNotFoundError:
    print("WARNING: Playlist data file not found")
    print("Creating empty placeholder file...")
    
    Path("data/processed").mkdir(exist_ok=True)
    pd.DataFrame(columns=['user_id', 'audio_id', 'playlist_add_score']).to_csv(
        "data/processed/playlist_signals.csv", index=False
    )
    print("Created empty: data/processed/playlist_signals.csv")

print("="*70 + "\n")