import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

import pandas as pd
import pickle

# ---------------- CONFIG ----------------
AUDIO_DATA_PATH = "data/raw/AudioShots_UAT.audios.csv"
AUDIO_LANG_CAT_PATH = "data/processed/audio_language_category.csv"
ML_INTERACTIONS_PATH = "data/processed/ml_interactions.csv"

ITEM_ENCODER_PATH = "models_saved/item_encoder.pkl"
LANGUAGE_ENCODER_PATH = "models_saved/language_encoder.pkl"
CATEGORY_ENCODER_PATH = "models_saved/category_encoder.pkl"

print("\n" + "="*80)
print("DIAGNOSTIC ANALYSIS - FINDING THE ROOT CAUSE")
print("="*80 + "\n")

# ---------------- STEP 1: Check Audio Metadata ----------------
print("STEP 1: CHECKING AUDIO METADATA")
print("-"*80)

audio_df = pd.read_csv(AUDIO_DATA_PATH)
print(f"✓ Total audios in raw data: {len(audio_df)}")
print(f"✓ Sample audio IDs from raw data:")
print(audio_df['_id'].head(10).tolist())
print()

# Check processed audio-language-category file
try:
    audio_lc_df = pd.read_csv(AUDIO_LANG_CAT_PATH)
    print(f"✓ Processed audio_language_category.csv has {len(audio_lc_df)} rows")
    print(f"✓ Columns: {list(audio_lc_df.columns)}")
    print(f"\n✓ Sample data:")
    print(audio_lc_df.head())
    print()
except Exception as e:
    print(f"✗ Error reading audio_language_category.csv: {e}")
    audio_lc_df = None

# ---------------- STEP 2: Check ML Interactions ----------------
print("\n" + "="*80)
print("STEP 2: CHECKING ML INTERACTIONS")
print("-"*80)

ml_df = pd.read_csv(ML_INTERACTIONS_PATH)
print(f"✓ Total interactions: {len(ml_df)}")
print(f"✓ Columns: {list(ml_df.columns)}")
print(f"✓ Unique items: {ml_df['item_idx'].nunique()}")
print(f"✓ Unique users: {ml_df['user_idx'].nunique()}")
print(f"✓ Unique languages: {ml_df['language_idx'].nunique()}")
print(f"✓ Unique categories: {ml_df['category_idx'].nunique()}")
print(f"\n✓ Sample ML interactions:")
print(ml_df.head(10))
print()

# ---------------- STEP 3: Check Encoders ----------------
print("\n" + "="*80)
print("STEP 3: CHECKING ENCODERS")
print("-"*80)

with open(ITEM_ENCODER_PATH, "rb") as f:
    item_encoder = pickle.load(f)

with open(LANGUAGE_ENCODER_PATH, "rb") as f:
    language_encoder = pickle.load(f)

with open(CATEGORY_ENCODER_PATH, "rb") as f:
    category_encoder = pickle.load(f)

print(f"✓ Item encoder has {len(item_encoder.classes_)} items")
print(f"✓ Sample encoded items (audio IDs):")
print(item_encoder.classes_[:10])
print()

print(f"✓ Language encoder has {len(language_encoder.classes_)} languages")
print(f"✓ Languages: {list(language_encoder.classes_)}")
print()

print(f"✓ Category encoder has {len(category_encoder.classes_)} categories")
print(f"✓ Categories: {list(category_encoder.classes_)}")
print()

# ---------------- STEP 4: Cross-check IDs ----------------
print("\n" + "="*80)
print("STEP 4: CROSS-CHECKING AUDIO IDs")
print("-"*80)

raw_audio_ids = set(audio_df['_id'].values)
encoded_audio_ids = set(item_encoder.classes_)

print(f"✓ Raw audio IDs: {len(raw_audio_ids)}")
print(f"✓ Encoded audio IDs: {len(encoded_audio_ids)}")

ids_in_both = raw_audio_ids.intersection(encoded_audio_ids)
ids_only_raw = raw_audio_ids - encoded_audio_ids
ids_only_encoded = encoded_audio_ids - raw_audio_ids

print(f"\n✓ Audio IDs in BOTH raw and encoder: {len(ids_in_both)}")
print(f"✗ Audio IDs ONLY in raw data: {len(ids_only_raw)}")
print(f"✗ Audio IDs ONLY in encoder: {len(ids_only_encoded)}")

if ids_only_raw:
    print(f"\n⚠ Sample IDs only in raw data (not encoded):")
    print(list(ids_only_raw)[:5])

if ids_only_encoded:
    print(f"\n⚠ Sample IDs only in encoder (not in raw):")
    print(list(ids_only_encoded)[:5])

# ---------------- STEP 5: Check Language-Category Mapping ----------------
print("\n" + "="*80)
print("STEP 5: CHECKING LANGUAGE-CATEGORY MAPPING")
print("-"*80)

if audio_lc_df is not None:
    # Build mapping from processed file
    from collections import defaultdict
    
    lang_cat_audio_map = defaultdict(lambda: defaultdict(set))
    
    for idx, row in audio_lc_df.iterrows():
        audio_id = row['audio_id']
        language = row['language']
        category = row['category']
        
        if pd.notna(audio_id) and pd.notna(language) and pd.notna(category):
            lang_cat_audio_map[language][category].add(audio_id)
    
    print(f"✓ Built mapping from audio_language_category.csv")
    print(f"✓ Languages found: {len(lang_cat_audio_map)}")
    
    for lang in sorted(lang_cat_audio_map.keys()):
        num_cats = len(lang_cat_audio_map[lang])
        total_audios = sum(len(audios) for audios in lang_cat_audio_map[lang].values())
        print(f"  • {lang}: {num_cats} categories, {total_audios} audio IDs")
    
    # Check if these audio IDs are in the encoder
    print(f"\n✓ Checking if mapped audio IDs exist in encoder...")
    
    mapped_audio_ids = set()
    for lang in lang_cat_audio_map:
        for cat in lang_cat_audio_map[lang]:
            mapped_audio_ids.update(lang_cat_audio_map[lang][cat])
    
    mapped_in_encoder = mapped_audio_ids.intersection(encoded_audio_ids)
    mapped_not_in_encoder = mapped_audio_ids - encoded_audio_ids
    
    print(f"✓ Mapped audio IDs: {len(mapped_audio_ids)}")
    print(f"✓ Mapped IDs in encoder: {len(mapped_in_encoder)} ({len(mapped_in_encoder)/len(mapped_audio_ids)*100:.1f}%)")
    print(f"✗ Mapped IDs NOT in encoder: {len(mapped_not_in_encoder)} ({len(mapped_not_in_encoder)/len(mapped_audio_ids)*100:.1f}%)")
    
    if mapped_not_in_encoder:
        print(f"\n⚠ Sample mapped IDs not in encoder:")
        print(list(mapped_not_in_encoder)[:5])

# ---------------- STEP 6: Test a Specific Language-Category ----------------
print("\n" + "="*80)
print("STEP 6: TESTING SPECIFIC LANGUAGE-CATEGORY COMBINATION")
print("-"*80)

TEST_LANG = "English"
TEST_CAT = "Psychology"

print(f"\nTest case: {TEST_LANG} → {TEST_CAT}")

# Check if these exist in encoders
try:
    lang_idx = language_encoder.transform([TEST_LANG])[0]
    print(f"✓ Language '{TEST_LANG}' encoded as index: {lang_idx}")
except:
    print(f"✗ Language '{TEST_LANG}' NOT found in encoder")

try:
    cat_idx = category_encoder.transform([TEST_CAT])[0]
    print(f"✓ Category '{TEST_CAT}' encoded as index: {cat_idx}")
except:
    print(f"✗ Category '{TEST_CAT}' NOT found in encoder")

# Check how many items in ML interactions have this language-category combo
if 'language_idx' in ml_df.columns and 'category_idx' in ml_df.columns:
    try:
        lang_idx = language_encoder.transform([TEST_LANG])[0]
        cat_idx = category_encoder.transform([TEST_CAT])[0]
        
        matching_interactions = ml_df[
            (ml_df['language_idx'] == lang_idx) & 
            (ml_df['category_idx'] == cat_idx)
        ]
        
        print(f"\n✓ ML interactions with {TEST_LANG} + {TEST_CAT}: {len(matching_interactions)}")
        
        if len(matching_interactions) > 0:
            unique_items = matching_interactions['item_idx'].nunique()
            print(f"✓ Unique items in these interactions: {unique_items}")
            
            # Show sample item indices
            sample_item_indices = matching_interactions['item_idx'].head(5).tolist()
            print(f"✓ Sample item indices: {sample_item_indices}")
            
            # Convert back to audio IDs
            sample_audio_ids = item_encoder.inverse_transform(sample_item_indices)
            print(f"✓ Corresponding audio IDs: {list(sample_audio_ids)}")
        else:
            print(f"✗ No interactions found for this combination!")
    except Exception as e:
        print(f"✗ Error: {e}")

# ---------------- STEP 7: Summary & Diagnosis ----------------
print("\n" + "="*80)
print("DIAGNOSIS SUMMARY")
print("="*80 + "\n")

print("🔍 KEY FINDINGS:")
print()

if len(ids_in_both) == 0:
    print("❌ CRITICAL ISSUE: No audio IDs match between raw data and encoder!")
    print("   → Your item_encoder was trained on different data")
    print("   → Need to rebuild encoders or check data preprocessing")
elif len(ids_in_both) < len(raw_audio_ids) * 0.5:
    print("⚠ WARNING: Only partial overlap between raw data and encoder")
    print(f"   → Only {len(ids_in_both)/len(raw_audio_ids)*100:.1f}% of raw audio IDs are in encoder")
else:
    print(f"✓ Good: {len(ids_in_both)/len(raw_audio_ids)*100:.1f}% of raw audio IDs are in encoder")

print()

if audio_lc_df is not None and mapped_not_in_encoder:
    print("⚠ WARNING: Some audio IDs in language-category mapping are not in encoder")
    print(f"   → {len(mapped_not_in_encoder)} mapped IDs ({len(mapped_not_in_encoder)/len(mapped_audio_ids)*100:.1f}%) missing from encoder")

print()
print("="*80)
print("\n💡 NEXT STEPS:")
print()
print("1. If audio IDs don't match: Rebuild the item_encoder using current audio data")
print("2. If language/category mapping is wrong: Check audio_language_category.csv")
print("3. If ML interactions don't have lang-cat combos: Check data preprocessing")
print()
print("="*80 + "\n")