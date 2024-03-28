import numpy as np
from sklearn.decomposition import TruncatedSVD from sklearn.metrics import
mean_squared_error from sklearn.model_selection import train_test_split from scipy.sparse
import coo_matrix
# Sample user-item matrix (replace this with your actual data)
# Rows represent users, columns represent items, and values represent ratings
user_item_matrix = np.array([[4, 5, 0, 0],
[0, 0, 5, 4],
[3, 0, 0, 5],
[0, 4, 0, 0]])
# Convert the user-item matrix to a sparse matrix for efficient computation
sparse_matrix = coo_matrix(user_item_matrix)
train_data, test_data = train_test_split(sparse_matrix, test_size=0.2,
random_state=42)
# Choose the number of latent factors (dimensions) n_latent_factors = 2
# Perform Truncated SVD
svd = TruncatedSVD(n_components=n_latent_factors) svd.fit(train_data)
# Transform the training and test data train_data_svd =
svd.transform(train_data)
test_data_svd = svd.transform(test_data)
# Reconstruct the matrix from the reduced dimensions predicted_ratings =
svd.inverse_transform(train_data_svd)
# Evaluate the performance using Mean Squared Error
mse = mean_squared_error(train_data.toarray(), predicted_ratings)
print(f"Mean Squared Error: {mse}")
# Make recommendations for a user (e.g., user 0) user_index = 0
user_predictions = svd.inverse_transform(test_data_svd)[user_index, :]
# Display the original and predicted ratings for the user print("Original
Ratings:") print(test_data.toarray()[user_index, :])
print("Predicted Ratings:") print(user_predictions)
