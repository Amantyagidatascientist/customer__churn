import pandas as pd
import numpy as np

# Parameters
num_users = 1000  # Number of unique users
num_movies = 500  # Number of unique movies
num_records = 10000  # Total records

# Generate user IDs
user_ids = np.random.choice(range(1, num_users + 1), size=num_records, replace=True)

# Generate movie IDs
movie_ids = np.random.choice(range(1, num_movies + 1), size=num_records, replace=True)

# Ensure each user watches at least 10 different movies
user_movie_map = {}
for user in range(1, num_users + 1):
    unique_movies = np.random.choice(range(1, num_movies + 1), size=10, replace=False)
    for movie in unique_movies:
        user_movie_map[(user, movie)] = np.random.randint(1, 6)  # Watch count between 1 and 5

# Generate watch count ensuring multiple users can watch the same movie multiple times
watch_counts = []
for user, movie in zip(user_ids, movie_ids):
    watch_counts.append(user_movie_map.get((user, movie), np.random.randint(1, 6)))

# Create DataFrame
df = pd.DataFrame({'user_id': user_ids, 'movie_id': movie_ids, 'watch_count': watch_counts})

# Save to CSV
csv_filename = "/mnt/data/user_movie_watch_data.csv"
df.to_csv(csv_filename, index=False)

csv_filename