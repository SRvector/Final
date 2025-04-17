import streamlit as st
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
from sklearn.cluster import KMeans
import plotly.express as px
import glob

st.set_page_config(page_title="Hashtag Trends Analyzer", layout="wide")
st.title("📈 Hashtag Trends Analyzer with Apriori & Clustering")

# Load all hashtag files
@st.cache_data
def load_data():
    files = glob.glob("hashtag_dataset_extracted/*_hashtag_trend_data.csv")
    if not files:
        st.error("No hashtag trend files found in 'hashtag_dataset_extracted'. Please check the directory.")
        return pd.DataFrame()
    all_data = []
    for file in files:
        try:
            df = pd.read_csv(file)
            df['source_file'] = file.split('/')[-1]
            all_data.append(df)
        except Exception as e:
            st.warning(f"Failed to read {file}: {e}")
    if not all_data:
        st.error("No valid data could be loaded.")
        return pd.DataFrame()
    return pd.concat(all_data, ignore_index=True)

combined_df = load_data()
if combined_df.empty:
    st.stop()

# Sidebar filters
country_filter = st.sidebar.multiselect("Select Country", options=combined_df['searched_in_country'].dropna().unique(), default=None)
date_filter = st.sidebar.date_input("Select Date Range", [])

if country_filter:
    combined_df = combined_df[combined_df['searched_in_country'].isin(country_filter)]

if len(date_filter) == 2:
    combined_df['searched_at_datetime'] = pd.to_datetime(combined_df['searched_at_datetime'])
    combined_df = combined_df[(combined_df['searched_at_datetime'] >= pd.to_datetime(date_filter[0])) &
                              (combined_df['searched_at_datetime'] <= pd.to_datetime(date_filter[1]))]

# Transactions for Apriori
transactions_df = combined_df.groupby(['searched_at_datetime', 'searched_in_country'])['trend_name'].apply(list).reset_index()
transactions = transactions_df['trend_name'].tolist()

te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df_encoded = pd.DataFrame(te_ary, columns=te.columns_)

# Apriori parameters
min_sup = st.sidebar.slider("Min Support", 0.01, 0.5, 0.05, 0.01)
min_conf = st.sidebar.slider("Min Confidence", 0.1, 1.0, 0.5, 0.05)

frequent_itemsets = apriori(df_encoded, min_support=min_sup, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_conf)

st.subheader("🔗 Association Rules")
st.dataframe(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

# Clustering on frequency and volume
hashtag_counts = combined_df['trend_name'].value_counts().reset_index()
hashtag_counts.columns = ['trend_name', 'frequency']
volumes = combined_df.groupby('trend_name')['tweet_volume'].mean().reset_index()
hashtag_stats = pd.merge(hashtag_counts, volumes, on='trend_name', how='left').fillna(0)

X = hashtag_stats[['frequency', 'tweet_volume']]
kmeans = KMeans(n_clusters=4, random_state=0).fit(X)
hashtag_stats['cluster'] = kmeans.labels_

# Clustering plot
st.subheader("📊 Clustering of Hashtags")
fig = px.scatter(hashtag_stats, x='frequency', y='tweet_volume', color='cluster', hover_data=['trend_name'])
st.plotly_chart(fig, use_container_width=True)

# Search box
search_term = st.text_input("Search for a specific hashtag")
if search_term:
    result = hashtag_stats[hashtag_stats['trend_name'].str.contains(search_term, case=False)]
    st.write(result)
