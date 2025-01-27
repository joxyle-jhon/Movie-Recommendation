import pandas as pd

# Define column names
columns = ['movie_id', 'title', 'release_date', 'video_release_date', 'IMDb_URL',
           'unknown', 'Action', 'Adventure', 'Animation', 'Children', 'Comedy',
           'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror',
           'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']

# Load the file
movies = pd.read_csv("ml-100k/u.item", sep="|", names=columns, encoding="latin-1", usecols=['movie_id', 'title', 'release_date'])

# Display first few rows
print(movies.head())


# Load ratings data
ratings = pd.read_csv("ml-100k/u.data", sep="\t", names=['user_id', 'movie_id', 'rating', 'timestamp'])

# Merge ratings with movie titles
data = pd.merge(ratings, movies, on="movie_id")

# Display merged dataset
print(data.head())

columns = ['movie_id', 'title', 'release_date', 'video_release_date', 'IMDb_URL', 
           'unknown', 'Action', 'Adventure', 'Animation', 'Children', 'Comedy', 
           'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 
           'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
movies = pd.read_csv("ml-100k/u.item", sep="|", names=columns, encoding='latin-1')

# Display first few rows
print(movies[['movie_id', 'title']].head())


# Merge ratings with movie titles
data = pd.merge(ratings, movies[['movie_id', 'title']], on="movie_id")

# Display merged data
print(data.head())

from surprise import Dataset, Reader
from surprise.model_selection import train_test_split
from surprise import SVD, accuracy

# Load data into Surprise
reader = Reader(rating_scale=(1, 5))  # Rating scale is from 1 to 5
data = Dataset.load_from_df(ratings[['user_id', 'movie_id', 'rating']], reader)

# Split data into training and test sets (80% train, 20% test)
trainset, testset = train_test_split(data, test_size=0.2)

# Display split data info
print(f"Training set: {len(trainset)} ratings")
print(f"Test set: {len(testset)} ratings")

# Create an SVD model
svd_model = SVD()

# Train the model on the training set
svd_model.fit(trainset)

# Make predictions on the test set
predictions = svd_model.test(testset)

# Calculate RMSE
rmse = accuracy.rmse(predictions)
print(f"RMSE: {rmse}")

def get_top_n_recommendations(predictions, n=10):
    # Convert predictions to a format {user: [(movie_id, rating_est), ...]}
    top_n = {}
    for uid, iid, true_r, est, _ in predictions:
        if uid not in top_n:
            top_n[uid] = []
        top_n[uid].append((iid, est))

    # Sort the predictions for each user and get the top N
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]
    
    return top_n

# Get top N recommendations for all users
top_n_recommendations = get_top_n_recommendations(predictions, n=10)

# Print top N recommendations for a specific user (e.g., user 1)
user_id = 1
recommended_movies = top_n_recommendations.get(user_id, [])
for movie_id, rating in recommended_movies:
    movie_title = movies[movies['movie_id'] == movie_id]['title'].values[0]
    print(f"Movie: {movie_title}, Predicted Rating: {rating}")
