# 1
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
# Sample data for items and their features items_data
= {
'Item1': [3, 4, 5],
'Item2': [1, 2, 3],
'Item3': [4, 5, 6],
'Item4': [2, 3, 4]
}
# Convert the dictionary to a matrix
items_matrix = np.array(list(items_data.values()))
# Calculate cosine similarity
similarity_matrix = cosine_similarity(items_matrix) print("Similarity Matrix:")
print(similarity_matrix)

# 3

class UserProfile:
def
init
(self):
self.preferences = {} # User preferences for items
def add_preference(self, item_id, rating): self.preferences[item_id] = rating
user_profiles = {} # Dictionary to store user profiles
# Example user profiles user_profiles['user1'] =
UserProfile() user_profiles['user2'] = UserProfile()
def update_user_profile(user_id, item_id, rating): if user_id in
user_profiles:
user_profile = user_profiles[user_id] user_profile.add_preference(item_id, rating)
else:
new_user_profile = UserProfile() new_user_profile.add_preference(item_id,
rating) user_profiles[user_id] = new_user_profile
feedback_data = [ ('user1','item1', 5),
('user1', 'item2', 4),
('user2', 'item1', 3),
('user2', 'item3', 5),
]
for user_id, item_id, rating in feedback_data: update_user_profile(user_id, item_id,
rating)
def recommend_items(user_id): if
user_id in user_profiles:
user_profile = user_profiles[user_id]
recommended_items = [item_id for item_id, rating in
user_profile.preferences.items() if rating >= 4]
return recommended_items else:
return [] user_id_to_recommend
= 'user1'
recommended_items = recommend_items(user_id_to_recommend)
print(f"Recommended items for {user_id_to_recommend}:
{recommended_items}")



#4
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer from sklearn.metrics.pairwise
import linear_kernel
data = {'ItemID': [1, 2, 3, 4],
'Description': ['Action movie with explosions', 'Romantic comedy with love story', 'Sci-fi adventure in space', 'Classic drama with intense emotions']}
df = pd.DataFrame(data)
tfidf_vectorizer = TfidfVectorizer(stop_words='english') tfidf_matrix =
tfidf_vectorizer.fit_transform(df['Description'])cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
def get_recommendations(item_id, cosine_sim=cosine_sim): idx = df[df['ItemID'] ==
item_id].index[0]
sim_scores = list(enumerate(cosine_sim[idx]))
sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
sim_scores = sim_scores[1:3] # Get the top 2 most similar items (excluding itself)
item_indices = [i[0] for i in sim_scores] return df['ItemID'].iloc[item_indices]
# Test the recommendation system item_id_to_recommend =1
recommendations = get_recommendations(item_id_to_recommend)
print(f"Recommendations for Item
{item_id_to_recommend}:")
print(df.loc[df['ItemID'].isin(recommendations), 'Description'].values)


# 5
import numpy as np import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
data = {
'User': ['User1', 'User2', 'User3', 'User4'],
'Item1': [5, 4, 3, 2],
'Item2': [4, 5, 2, 3],
'Item3': [2, 3, 5, 4],
}
df = pd.DataFrame(data) df.set_index('User', inplace=True)
user_similarity = cosine_similarity(df)
np.fill_diagonal(user_similarity, 0) # Set diagonal elements to 0 to avoid self-similarity
user_similarity_df = pd.DataFrame(user_similarity, index=df.index, columns=df.index)def predict_rating(user, item):
numerator = np.sum(user_similarity_df[user] * df[item]) denominator =
np.sum(np.abs(user_similarity_df[user])) if denominator == 0:
return 0
return numerator / denominator
def recommend_items(user):
unrated_items = df.columns[df.loc[user] == 0]
predictions = [predict_rating(user, item) for item in unrated_items]
recommendations = pd.DataFrame({'Item': unrated_items, 'Prediction': predictions})
return recommendations.sort_values(by='Prediction', ascending=False)
user_to_recommend = 'User1'
recommendations = recommend_items(user_to_recommend) print(f"Recommendations for
{user_to_recommend}:") print(recommendations)


# 7
import numpy as np import pandas as pd
from sklearn.metrics import roc_curve, auc import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split from sklearn.ensemble import
RandomForestClassifier
# Generate synthetic data np.random.seed(42)
user_ids = np.repeat(np.arange(1, 101), 10)
item_ids = np.tile(np.arange(1, 11), 100)
ratings = np.random.choice([0, 1], size=len(user_ids), p=[0.8, 0.2])
# Create a DataFrame
data = pd.DataFrame({'user_id': user_ids, 'item_id': item_ids, 'rating': ratings})train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
# For simplicity, we use a RandomForestClassifier as an example
classifier = RandomForestClassifier(random_state=42) X_train =
pd.get_dummies(train_data[['user_id', 'item_id']]) y_train = train_data['rating']
classifier.fit(X_train, y_train)
# Predict probabilities for the test set
X_test = pd.get_dummies(test_data[['user_id', 'item_id']]) y_score =
classifier.predict_proba(X_test)[:, 1]
# Compute ROC curve and ROC area for each class fpr, tpr, _ = roc_curve(test_data['rating'],
y_score) roc_auc = auc(fpr, tpr)
# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--') plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve') plt.legend(loc='lower right')
plt.show()

