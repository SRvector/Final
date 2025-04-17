
# Hashtag Trend Analysis with Apriori & Clustering

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
from sklearn.cluster import KMeans
import plotly.express as px
import matplotlib.pyplot as plt

# Load combined CSVs
import glob
files = glob.glob("hashtag_dataset_extracted/*_hashtag_trend_data.csv")
all_data = []
for file in files:
    df = pd.read_csv(file)
    df['source_file'] = file.split('/')[-1]
    all_data.append(df)
combined_df = pd.concat(all_data, ignore_index=True)

# Convert to transactions
transactions_df = combined_df.groupby(['searched_at_datetime', 'searched_in_country'])['trend_name'].apply(list).reset_index()
transactions = transactions_df['trend_name'].tolist()

# One-hot encode
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df_encoded = pd.DataFrame(te_ary, columns=te.columns_)

# Apriori Algorithm
frequent_itemsets = apriori(df_encoded, min_support=0.02, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.4)
print("Association Rules:")
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head())

# Clustering
hashtag_counts = combined_df['trend_name'].value_counts().reset_index()
hashtag_counts.columns = ['trend_name', 'frequency']
volumes = combined_df.groupby('trend_name')['tweet_volume'].mean().reset_index()
hashtag_stats = pd.merge(hashtag_counts, volumes, on='trend_name', how='left').fillna(0)

X = hashtag_stats[['frequency', 'tweet_volume']]
kmeans = KMeans(n_clusters=4, random_state=0).fit(X)
hashtag_stats['cluster'] = kmeans.labels_

# Plot clusters
fig = px.scatter(hashtag_stats, x='frequency', y='tweet_volume', color='cluster', hover_data=['trend_name'])
fig.show()
