import pandas as pd

print("Loading raw dataset...")

columns = ['sentiment', 'id', 'date', 'query', 'user', 'text']

df = pd.read_csv(
    'data/raw/training.1600000.processed.noemoticon.csv',
    encoding='latin-1',
    header=None,
    names=columns
)

print("Original rows:", len(df))

# Convert labels
df['sentiment'] = df['sentiment'].map({0: 0, 4: 1})

df = df[['text', 'sentiment']]

# Balance classes
neg = df[df.sentiment == 0].sample(50000, random_state=42)
pos = df[df.sentiment == 1].sample(50000, random_state=42)

df = pd.concat([neg, pos])
df = df.sample(frac=1, random_state=42)

print("Balanced rows:", len(df))

df.to_csv("data/processed/sentiment_data.csv", index=False)

print("Saved balanced dataset")
