import streamlit as st
import pandas as pd
import numpy as np
import re
import csv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pymongo import MongoClient


uri = "mongodb+srv://pushpanjali:Pushpanjali123@blogplatform.3q8rv.mongodb.net/"
client = MongoClient(uri)

# Create CSV files from MongoDB collections
def createCSV(df):
    documents = df.find()
    csv_file_name = f"./data/{df.name}.csv"
    keys = documents[0].keys() if documents else []
    with open(csv_file_name, mode='w', newline='', encoding='utf-8') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=keys)
        writer.writeheader()
        for document in documents:
            writer.writerow(document)

createCSV(client.Rishit.blogs)
createCSV(client.Rishit.users)
createCSV(client.Rishit.comments)
createCSV(client.Rishit.interactions)


blogs = pd.read_csv('./data/blogs.csv')


def clean(text):
    return (re.sub("[^a-zA-Z0-9 ]", "", text) + " ")

blogs["tokens"] = blogs["title"].apply(clean) + blogs["content"].apply(clean) + blogs["category"].apply(clean)
blogs["tokens"] = blogs["tokens"].str.lower()


tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(blogs['tokens'])


def suggest(text):
    text = clean(text)
    query_vec = tfidf_vectorizer.transform([text])
    similarity = cosine_similarity(query_vec, tfidf_matrix).flatten()
    indices = np.argpartition(similarity, -5)[-5:]
    results = blogs.iloc[indices].iloc[::-1]
    return results[['title', 'category']]

# Streamlit 
def main():
    st.title("Blog Suggestion System")
    
    user_input = st.text_input("Enter any text for blog suggestions:")
    
    if user_input:
        suggestions = suggest(user_input)
        
        if not suggestions.empty:
            st.write("Top 5 most similar blogs:")
            for index, row in suggestions.iterrows():
                st.write(f"Title: {row['title']}, Category: {row['category']}")
        else:
            st.write("No similar blogs found.")

if __name__ == "__main__":
    main()
