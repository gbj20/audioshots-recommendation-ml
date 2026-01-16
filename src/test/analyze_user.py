"""
Analyze User Interaction Distribution
Run this to understand why you're losing users
"""

import pandas as pd
import numpy as np

print("\n" + "="*80)
print("USER INTERACTION DISTRIBUTION ANALYSIS")
print("="*80 + "\n")

# Load raw data
print("STEP 1: RAW DATA")
print("-"*80)
users_raw = pd.read_csv("data/raw/AudioShots_UAT.users.csv")
likes_raw = pd.read_csv("data/raw/AudioShots_UAT.likes.csv")
progress_raw = pd.read_csv("data/raw/AudioShots_UAT.listeningprogresses.csv")

print(f"Total users in raw data: {len(users_raw)}")
print(f"Total likes: {len(likes_raw)}")
print(f"Total listening progress records: {len(progress_raw)}\n")

# Find user column names
user_col_likes = [c for c in likes_raw.columns if 'user' in c.lower() and 'id' in c.lower()][0]
user_col_progress = [c for c in progress_raw.columns if 'user' in c.lower() and 'id' in c.lower()][0]

print(f"User column in likes: {user_col_likes}")
print(f"User column in progress: {user_col_progress}\n")

# Count interactions per user
likes_users = likes_raw[user_col_likes].value_counts()
progress_users = progress_raw[user_col_progress].value_counts()

# Combine
all_user_ids = set(likes_raw[user_col_likes].unique()) | set(progress_raw[user_col_progress].unique())

interaction_counts = {}
for uid in all_user_ids:
    if pd.notna(uid):
        count = likes_users.get(uid, 0) + progress_users.get(uid, 0)
        interaction_counts[uid] = count

# Create distribution
counts_df = pd.DataFrame(list(interaction_counts.items()), columns=['user_id', 'interaction_count'])
counts_df = counts_df.sort_values('interaction_count', ascending=False)

print("STEP 2: INTERACTION DISTRIBUTION")
print("-"*80)
print(f"Users with any interaction: {len(counts_df)}")
print(f"Users with NO interactions: {len(users_raw) - len(counts_df)}\n")

print("Distribution by interaction count:")
print("-"*80)
bins = [0, 1, 3, 5, 10, 20, 50, 100, float('inf')]
labels = ['0', '1-2', '3-4', '5-9', '10-19', '20-49', '50-99', '100+']

counts_df['bin'] = pd.cut(counts_df['interaction_count'], bins=bins, labels=labels, right=False)
distribution = counts_df['bin'].value_counts().sort_index()

for bin_label, count in distribution.items():
    percentage = (count / len(counts_df)) * 100
    print(f"  {bin_label:8s}: {count:4d} users ({percentage:5.1f}%)")

print()

# Check each threshold
print("STEP 3: IMPACT OF DIFFERENT THRESHOLDS")
print("-"*80)
for threshold in [1, 2, 3, 5, 10]:
    kept = (counts_df['interaction_count'] >= threshold).sum()
    lost = len(counts_df) - kept
    percentage = (kept / len(counts_df)) * 100
    print(f"  Threshold >= {threshold:2d}: Keep {kept:3d} users ({percentage:5.1f}%), Lose {lost:3d}")

print()

# Show top users
print("STEP 4: TOP 10 MOST ACTIVE USERS")
print("-"*80)
for i, row in counts_df.head(10).iterrows():
    print(f"  {row['user_id']}: {row['interaction_count']} interactions")

print()

# Show users right at the threshold
print("STEP 5: USERS WITH EXACTLY 1-3 INTERACTIONS (Currently Being Filtered)")
print("-"*80)
low_activity = counts_df[(counts_df['interaction_count'] >= 1) & (counts_df['interaction_count'] < 3)]
print(f"Users with 1-2 interactions: {len(low_activity)}")
if len(low_activity) > 0:
    print("\nSample of these users:")
    for i, row in low_activity.head(5).iterrows():
        print(f"  {row['user_id']}: {row['interaction_count']} interactions")

print()

print("="*80)
print("RECOMMENDATION")
print("="*80)

users_with_1_plus = (counts_df['interaction_count'] >= 1).sum()
users_with_3_plus = (counts_df['interaction_count'] >= 3).sum()
increase = users_with_1_plus - users_with_3_plus

print(f"Current setting (threshold >= 3): {users_with_3_plus} users")
print(f"If changed to (threshold >= 1): {users_with_1_plus} users")
print(f"Would add {increase} more users to training data\n")

if increase > 50:
    print("✓ RECOMMENDED: Change threshold to >= 1 to include more users")
    print("  This will significantly improve diversity")
elif increase > 20:
    print("⚠ CONSIDER: Changing threshold to >= 1 would help diversity")
else:
    print("⚠ WARNING: Even with threshold=1, you have limited users")
    print("  Consider collecting more user interaction data")

print()
print("="*80 + "\n")