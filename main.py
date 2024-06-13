import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)

movies_data = pd.read_csv('C:/Users/84199/Downloads/IMDB Top 250 Movies.csv')
movies_data.head()

print("Data Size:", movies_data.shape)
print("-" * 30)
print("About Dataset:")
movies_data.info()
print("-" * 30)
print("Data Columns:", list(movies_data.columns))
print("-" * 30)
print("Number of Examples 'N'=", movies_data.shape[0])
print("Number of Dimensions 'D'=", movies_data.shape[1] - 1)
print("-" * 30)
print("Data Check For Any Duplicates:", movies_data.duplicated().any())

movies_data.describe()

na_counts = movies_data.apply(lambda x: x.value_counts().get('Not Available', 0))
print('Not Available Values:')
print(na_counts)

movies_data['budget'] = movies_data['budget'].str.replace('Not Available', '0')
movies_data['budget'] = movies_data['budget'].apply(lambda x: float(''.join(filter(str.isdigit, str(x)))))
movies_data['box_office'] = movies_data['box_office'].str.replace('Not Available', '0')
movies_data['box_office'] = movies_data['box_office'].apply(lambda x: float(''.join(filter(str.isdigit, str(x)))))
movies_data.drop(['rank'], axis=1, inplace=True)

movies_data['run_time'] = movies_data['run_time'].replace('Not Available', pd.NaT)


def convert_to_timedelta(duration_str):
    duration_str = duration_str.replace('h', ' hours ').replace('m', ' minutes')
    duration = pd.to_timedelta(duration_str)
    return duration


movies_data['run_time'] = movies_data['run_time'].apply(convert_to_timedelta)

movies_data.describe()

numerical_data = movies_data.select_dtypes(include='number')

for column in numerical_data:
    print("'" + column + "'", "Range Of Values:", numerical_data[column].min(), "-", numerical_data[column].max())
    print("-" * 30)
print("Values Shown Previously Prove That We Need To Re-Scale Some of Our Features' Values.")

corr = round(movies_data.corr(), 2)
sns.heatmap(corr, annot=True, cmap='viridis')

z_scores = numerical_data[numerical_data.columns[1:]].apply(lambda x: (x - x.mean()) / x.std())
outliers = (z_scores > 3) | (z_scores < -3)
print(outliers.sum())

fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 5))
axs = axs.ravel()
fig.subplots_adjust(hspace=0.5, wspace=0.5)

for i, col in enumerate(numerical_data.columns[1:2].append(numerical_data.columns[3:])):
    sns.boxplot(y=col, data=numerical_data, ax=axs[i], boxprops={'color': 'green', 'edgecolor': 'black'},
                medianprops={'color': 'red'})
    axs[i].set_ylabel(col)
    axs[i].set_xlabel('Count')

value_counts = numerical_data['rating'].value_counts()
print(value_counts[value_counts > 1])

top_movies_by_genre = movies_data.groupby("genre").apply(lambda x: x.loc[x["rating"].idxmax()])
top_movies_by_genre[["rating", "name"]].head()

# Plot the distribution of ratings using a histogram
plt.hist(movies_data["rating"], bins=20, edgecolor='black', color='green')
plt.xlabel("Rating")
plt.ylabel("Number of Movies")
plt.title("Distribution of Ratings in IMDB Top 250 Movies Dataset")
plt.show()

director_ratings = movies_data.groupby("directors")["rating"].mean()
director_ratings = director_ratings.sort_values(ascending=False)

top_directors = director_ratings.head(10)
plt.barh(top_directors.index, top_directors.values, edgecolor='black', color='green')
plt.xlabel("Average Rating")
plt.ylabel("Director")
plt.title("Top 10 Directors by Average Rating in IMDB Top 250 Movies Dataset")
plt.show()

casts_ratings = movies_data.groupby("casts")["rating"].mean()
casts_ratings = casts_ratings.sort_values(ascending=False)
top_casts = casts_ratings.head(10)
shortened_labels = [name[:10] + '...' if len(name) > 10 else name for name in top_casts.index]
plt.barh(top_casts.index, top_casts.values, edgecolor='black', color='green')
plt.yticks(top_casts.index, shortened_labels)
plt.xlabel("Average Rating")
plt.ylabel("Casts")
plt.title("Top 10 Casts by Average Rating in IMDB Top 250 Movies Dataset")
plt.show()

director_movies = movies_data.groupby("directors")["name"].apply(list)
vectorizer = TfidfVectorizer()
movie_vectors = vectorizer.fit_transform(movies_data["name"])
cosine_sim = cosine_similarity(movie_vectors)


def recommend_directors(movie_title):
    movie_index = movies_data[movies_data["name"] == movie_title].index[0]
    director_similarities = {}
    for director, movies in director_movies.items():
        movie_indices = [movies_data[movies_data["name"] == title].index[0] for title in movies]
        director_similarities[director] = cosine_sim[movie_index, movie_indices].mean()
    top_directors = sorted(director_similarities, key=director_similarities.get, reverse=True)[:5]
    return top_directors


directros_recommended = recommend_directors("The Godfather")
rank = 0
print('The 5 highest directors recommended to watch their movies, with a similarity of "The Godfather" movie are:')
for director in directros_recommended:
    rank += 1
    print(str(rank) + ".", director)

dict_genres = {}
for genres in movies_data.genre:
    genre_list = genres.split(',')
    for g in genre_list:
        genre = g.strip()
        if genre not in dict_genres:
            dict_genres[genre] = 1
        else:
            dict_genres[genre] += 1

print(dict_genres)

year_ratings = movies_data.groupby("year")["rating"].mean()
plt.plot(year_ratings.index, year_ratings.values, color='purple')
plt.xlabel("Year")
plt.ylabel("Average Rating")
plt.title("Year-wise Analysis of Top-rated Movies in IMDB Top 250 Movies Dataset")
plt.show()

movies_data.drop(['casts', 'directors', 'writers'], axis=1, inplace=True)
scaler = MinMaxScaler()
scaler.fit(movies_data[['budget', 'box_office']])
scaled_data = scaler.transform(movies_data[['budget', 'box_office']])

genres = movies_data['genre'].tolist()
certificates = movies_data['certificate'].tolist()
taglines = movies_data['tagline'].tolist()

tfidf_genres = TfidfVectorizer()
tfidf_certificate = TfidfVectorizer()
tfidf_tagline = TfidfVectorizer()

tfidf_genres.fit(genres)
tfidf_certificate.fit(certificates)
tfidf_tagline.fit(taglines)

feature_names_genres = tfidf_genres.get_feature_names_out()
feature_names_certificates = tfidf_certificate.get_feature_names_out()
feature_names_taglines = tfidf_tagline.get_feature_names_out()


def Text_Score(text, tfidf, feature_names):
    tfidf_matrix = tfidf.transform([text]).todense()
    feature_index = tfidf_matrix[0, :].nonzero()[1]
    tfidf_scores = zip([feature_names[i] for i in feature_index], [tfidf_matrix[0, x] for x in feature_index])
    text_dict = dict(tfidf_scores)
    text_scores = text_dict.values()
    return sum(text_scores)


movies_data['genre'] = movies_data['genre'].apply(lambda x: Text_Score(x, tfidf_genres, feature_names_genres))
movies_data['certificate'] = movies_data['certificate'].apply(
    lambda x: Text_Score(x, tfidf_certificate, feature_names_certificates))
movies_data['tagline'] = movies_data['tagline'].apply(lambda x: Text_Score(x, tfidf_tagline, feature_names_taglines))

movies_data.head()

movies_data['run_time'].fillna(movies_data['run_time'].mean(), inplace=True)
movies_data['run_time_minutes'] = movies_data['run_time'].dt.total_seconds() / 60

movies_data.drop('run_time', axis=1, inplace=True)
X = movies_data.drop(['year', 'rating', 'name'], axis=1)
y = movies_data['rating']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

rf = RandomForestRegressor()
rf.fit(X_train, y_train)

importances = rf.feature_importances_

feature_importances = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
feature_importances = feature_importances.sort_values(by='Importance', ascending=False)

plt.barh(feature_importances['Feature'], feature_importances['Importance'], edgecolor='black', color='green')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importance: IMDB Top 250 Movies')
plt.show()