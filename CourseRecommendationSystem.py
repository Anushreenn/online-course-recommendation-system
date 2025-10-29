# courserecommendationsystem.py

import os
import numpy as np
import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem.porter import PorterStemmer

print("🔹 Dependencies Imported")

# --- Load Dataset ---
data_path = "Data/Coursera.csv"
if not os.path.exists(data_path):
    raise FileNotFoundError("❌ Coursera.csv not found in 'Data' folder!")

data = pd.read_csv(data_path)
print(f"✅ Dataset Loaded: {data.shape[0]} courses and {data.shape[1]} columns")

# --- Basic Data Cleaning ---
cols_needed = ['Course Name', 'Difficulty Level', 'Course Description', 'Skills', 'Course URL']
existing_cols = [c for c in cols_needed if c in data.columns]
data = data[existing_cols].copy()

data.fillna('', inplace=True)

for col in ['Course Name', 'Course Description', 'Skills']:
    data[col] = data[col].astype(str).str.replace('[^a-zA-Z0-9\s]', ' ', regex=True)
    data[col] = data[col].str.replace('\s+', ' ', regex=True)

# --- Combine Tags ---
data['tags'] = (data['Course Name'] + ' ' +
                data['Difficulty Level'] + ' ' +
                data['Course Description'] + ' ' +
                data['Skills'])

# --- Lowercase + Stemming ---
ps = PorterStemmer()

def stem(text):
    return " ".join([ps.stem(i) for i in text.lower().split()])

data['tags'] = data['tags'].apply(stem)

# --- New DataFrame for Modeling ---
new_df = data[['Course Name', 'tags']].rename(columns={'Course Name': 'course_name'})

# --- Vectorization ---
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(new_df['tags']).toarray()

# --- Similarity Matrix ---
similarity = cosine_similarity(vectors)

# --- Save Models ---
os.makedirs('models', exist_ok=True)
pickle.dump(similarity, open('models/similarity.pkl', 'wb'))
pickle.dump(new_df, open('models/courses.pkl', 'wb'))

print("✅ Models Exported Successfully to 'models/' folder")

# --- Optional Test ---
def recommend(course):
    if course not in new_df['course_name'].values:
        return []
    index = new_df[new_df['course_name'] == course].index[0]
    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
    return [new_df.iloc[i[0]].course_name for i in distances[1:6]]

print("🔹 Example Recommendation:")
print(recommend(new_df['course_name'].iloc[0]))
