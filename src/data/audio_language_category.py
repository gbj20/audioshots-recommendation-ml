import pandas as pd

audio_df = pd.read_csv("C:\\Users\\gayat\\OneDrive\\Desktop\\audioshots-recommendation-ml\\audioshots-recommendation-ml\\data\\raw\\AudioShots_UAT.audios.csv")

rows = []

for _, row in audio_df.iterrows():
    audio_id = row["_id"]
    title = row["title"]
    language = row["language.name"]

    for i in range(5):  # categories[0] to categories[4]
        cat_col = f"categories[{i}].name"
        if cat_col in audio_df.columns and pd.notna(row[cat_col]):
            rows.append({
                "audio_id": audio_id,
                "title": title,
                "language": language,
                "category": row[cat_col]
            })

audio_lc_df = pd.DataFrame(rows)
audio_lc_df.to_csv("C:\\Users\\gayat\\OneDrive\\Desktop\\audioshots-recommendation-ml\\audioshots-recommendation-ml\\data\\processed\\audio_language_category.csv", index=False)

print(audio_lc_df.head())
