import pandas as pd

# Load data
likes = pd.read_csv("data/raw/AudioShots_UAT.likes.csv")
listens = pd.read_csv("data/raw/AudioShots_UAT.listeningprogresses.csv")

# Rename columns to ML standard
likes = likes.rename(columns={
    "userId": "user_id",
    "audioId": "audio_id"
})

listens = listens.rename(columns={
    "userId": "user_id",
    "audioId": "audio_id"
})

# Explicit feedback: like = strong signal
likes["score"] = 3.0

# Implicit feedback: normalize listening duration per user
listens["score"] = listens.groupby("user_id")["listeningDuration"].transform(
    lambda x: x / x.max()
)

# Combine interactions
interactions = pd.concat([
    likes[["user_id", "audio_id", "score"]],
    listens[["user_id", "audio_id", "score"]]
])

# Aggregate multiple interactions
interactions = interactions.groupby(
    ["user_id", "audio_id"], as_index=False
).sum()

# Save processed interactions
interactions.to_csv("data/processed/interactions.csv", index=False)

print("interactions.csv created successfully")
