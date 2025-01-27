# Movie Recommendation System

This project implements a **Movie Recommendation System** that suggests movies to users based on their preferences. It uses collaborative filtering with **Singular Value Decomposition (SVD)** to predict ratings and recommend movies that users might like. The dataset used for this project is the **MovieLens 100k dataset**, which contains user ratings and movie information.

## Features

- **Collaborative Filtering**: Recommendations are based on similar user preferences.
- **SVD (Singular Value Decomposition)**: A matrix factorization technique used for building the recommendation system.
- **Movie Recommendations**: Recommend movies to users based on their past ratings and the ratings of similar users.
- **Evaluation**: The system is evaluated using **Root Mean Squared Error (RMSE)** to measure prediction accuracy.

## Dataset

The dataset used in this project is the **MovieLens 100k dataset** available from [Kaggle](https://www.kaggle.com/datasets/prajitdatta/movielens-100k-dataset/data).

The dataset includes the following files:
- **u.data**: Contains user ratings for movies.
- **u.item**: Contains movie metadata (movie ID, title, genre, etc.).
- **u.user**: Contains user information (user ID, age, gender, etc.).

## Installation

### Prerequisites

- Python 3.x
- Libraries: `pandas`, `numpy`, `scikit-learn`, `surprise`, `matplotlib`

### Install Dependencies

```bash
pip install pandas numpy scikit-learn surprise matplotlib
