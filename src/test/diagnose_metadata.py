"""
Diagnose why some items are missing from metadata
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

import pandas as pd
import pickle

print("\n" + "="*70)
print("DIAGNOSING METADATA ISSUE")
print("="*70 + "\n")

# Load data
audio_df = pd.read_csv("data/processed/audio_language_category.csv")
with open("models_saved/item_encoder.pkl", "rb") as f:
    item_encoder = pickle.load(f)

print(f"✓ Audio metadata rows: {len(audio_df)}")
print(f"✓ Unique audio IDs in metadata: {audio_df['audio_id'].nunique()}")
print(f"✓ Items in encoder: {len(item_encoder.classes_)}")

# Check which items are in encoder but not in audio_df
audio_ids_in_metadata = set(audio_df['audio_id'].unique())
audio_ids_in_encoder = set(item_encoder.classes_)

in_encoder_not_metadata = audio_ids_in_encoder - audio_ids_in_metadata
in_metadata_not_encoder = audio_ids_in_metadata - audio_ids_in_encoder
in_both = audio_ids_in_encoder.intersection(audio_ids_in_metadata)

print(f"\n✓ Audio IDs in BOTH: {len(in_both)}")
print(f"✗ In encoder but NOT metadata: {len(in_encoder_not_metadata)}")
print(f"✗ In metadata but NOT encoder: {len(in_metadata_not_encoder)}")

if in_encoder_not_metadata:
    print(f"\n⚠ Sample IDs in encoder but missing metadata:")
    for aid in list(in_encoder_not_metadata)[:5]:
        print(f"  • {aid}")

# Check specific problem categories
print(f"\n{'='*70}")
print("CHECKING PROBLEM CATEGORIES")
print(f"{'='*70}\n")

problem_combos = [
    ("English", "Educational"),
    ("English", "Neurobiology"),
    ("English", "Startup"),
]

for lang, cat in problem_combos:
    # Count in metadata
    count_metadata = len(audio_df[
        (audio_df['language'] == lang) & 
        (audio_df['category'] == cat)
    ])
    
    # Get audio IDs
    audio_ids = audio_df[
        (audio_df['language'] == lang) & 
        (audio_df['category'] == cat)
    ]['audio_id'].unique()
    
    # Check how many are in encoder
    in_encoder = sum(1 for aid in audio_ids if aid in item_encoder.classes_)
    
    print(f"{lang} → {cat}:")
    print(f"  Metadata count: {count_metadata}")
    print(f"  Unique IDs: {len(audio_ids)}")
    print(f"  IDs in encoder: {in_encoder}")
    print()

print("="*70 + "\n")