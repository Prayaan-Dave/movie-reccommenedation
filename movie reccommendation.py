
import os
import random
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob
from colorama import init, Fore, Style

# Initialize colors
init(autoreset=True)

# ==========================================
#          AI MOVIE RECOMMENDER
# ==========================================

APP_NAME = "AI MOVIE RECOMMENDATION SYSTEM"

DATASET = "imdb_top_1000.csv"

# Mood descriptions used by the AI
MOODS = {
    "happy": "funny cheerful uplifting joyful comedy",
    "sad": "emotional heartbreaking dramatic tragic",
    "excited": "action adventure thrilling exciting suspense",
    "scared": "horror frightening mysterious terrifying",
    "romantic": "romance love relationships emotional",
    "relaxed": "peaceful calm relaxing heartwarming",
    "curious": "mystery investigation detective secrets",
    "inspired": "motivational inspirational success courage"
}


# ==========================================
#             HELPER FUNCTIONS
# ==========================================

def print_line():
    print(Fore.CYAN + "=" * 65)


def welcome_screen():
    print("\n")
    print_line()
    print(Fore.YELLOW + Style.BRIGHT + APP_NAME.center(65))
    print_line()

    print(Fore.GREEN + "\nWelcome to your personal AI movie assistant!")
    print("Discover movies based on your interests, mood, and ratings.")
    print()


def load_movies():

    if not os.path.exists(DATASET):
        print(Fore.RED + "\nERROR: Dataset not found!")
        print("Make sure imdb_top_1000.csv is inside the project folder.")
        return None

    try:
        movies = pd.read_csv(DATASET, encoding="utf-8-sig")

    except Exception as error:
        print(Fore.RED + f"\nCould not load dataset: {error}")
        return None

    # Normalize column names
    movies.columns = [
        str(column).strip().lower().replace(" ", "_")
        for column in movies.columns
    ]

    # Support common alternative column names
    aliases = {
        "title": "series_title",
        "movie_title": "series_title",
        "rating": "imdb_rating",
        "overview": "overview",
        "description": "overview",
        "genre": "genre"
    }

    for old_name, new_name in aliases.items():
        if old_name in movies.columns and new_name not in movies.columns:
            movies.rename(columns={old_name: new_name}, inplace=True)

    required_columns = [
        "series_title",
        "genre",
        "overview",
        "imdb_rating"
    ]

    missing = [
        column for column in required_columns
        if column not in movies.columns
    ]

    if missing:
        print(Fore.RED + "\nDataset is missing required columns:")
        print(", ".join(missing))
        return None

    # Clean movie information
    movies = movies.dropna(
        subset=["series_title", "genre", "overview", "imdb_rating"]
    ).copy()

    movies["imdb_rating"] = pd.to_numeric(
        movies["imdb_rating"],
        errors="coerce"
    )

    movies = movies.dropna(subset=["imdb_rating"])

    movies["series_title"] = movies["series_title"].astype(str)
    movies["genre"] = movies["genre"].astype(str)
    movies["overview"] = movies["overview"].astype(str)

    movies = movies.reset_index(drop=True)

    # Combine genres and overviews for AI processing
    movies["combined_features"] = (
        movies["genre"] + " " +
        movies["overview"]
    )

    return movies


# ==========================================
#           SENTIMENT ANALYSIS
# ==========================================

def analyze_sentiment(overview):

    analysis = TextBlob(overview)
    polarity = analysis.sentiment.polarity

    if polarity > 0.05:
        return "Positive", Fore.GREEN

    elif polarity < -0.05:
        return "Negative", Fore.RED

    else:
        return "Neutral", Fore.YELLOW


# ==========================================
#             GENRE FILTER
# ==========================================

def get_genres(movies):

    all_genres = set()

    for genres in movies["genre"]:
        for genre in genres.split(","):
            all_genres.add(genre.strip())

    return sorted(all_genres)


def choose_genre(movies):

    genres = get_genres(movies)

    print(Fore.YELLOW + "\nAVAILABLE GENRES")
    print("-" * 40)

    for genre in genres:
        print(Fore.CYAN + genre)

    print(Fore.WHITE + "\nType a genre or press ENTER for any genre.")

    while True:

        choice = input(Fore.GREEN + "Your genre: ").strip()

        if choice == "":
            return None

        matching = [
            genre for genre in genres
            if choice.lower() == genre.lower()
        ]

        if matching:
            return matching[0]

        print(Fore.RED + "Genre not found. Please try again.")


# ==========================================
#              MOOD FILTER
# ==========================================

def choose_mood():

    print(Fore.YELLOW + "\nHOW ARE YOU FEELING?")
    print("-" * 40)

    available_moods = list(MOODS.keys())

    for number, mood in enumerate(available_moods, start=1):
        print(Fore.CYAN + f"{number}. {mood.title()}")

    print(Fore.CYAN + "0. Any mood")

    while True:

        choice = input(Fore.GREEN + "\nChoose your mood: ").strip()

        if choice == "0" or choice == "":
            return None

        if choice.isdigit():

            number = int(choice)

            if 1 <= number <= len(available_moods):
                return available_moods[number - 1]

        matching = [
            mood for mood in available_moods
            if mood.lower() == choice.lower()
        ]

        if matching:
            return matching[0]

        print(Fore.RED + "Invalid mood. Try again.")


# ==========================================
#             RATING FILTER
# ==========================================

def choose_rating():

    print(Fore.YELLOW + "\nMINIMUM IMDb RATING")
    print("Choose a minimum rating from 0 to 10.")
    print("Press ENTER for no minimum rating.")

    while True:

        choice = input(Fore.GREEN + "Minimum rating: ").strip()

        if choice == "":
            return 0.0

        try:
            rating = float(choice)

            if 0 <= rating <= 10:
                return rating

        except ValueError:
            pass

        print(Fore.RED + "Enter a number between 0 and 10.")


# ==========================================
#             FILTER MOVIES
# ==========================================

def filter_movies(movies, genre, minimum_rating):

    filtered = movies.copy()

    if genre:
        filtered = filtered[
            filtered["genre"].str.contains(
                genre,
                case=False,
                regex=False,
                na=False
            )
        ]

    if minimum_rating > 0:
        filtered = filtered[
            filtered["imdb_rating"] >= minimum_rating
        ]

    return filtered


# ==========================================
#            AI RECOMMENDATION
# ==========================================

def ai_recommendation(
    movies,
    genre,
    mood,
    minimum_rating,
    already_recommended
):

    filtered = filter_movies(
        movies,
        genre,
        minimum_rating
    )

    # Remove previously recommended movies
    filtered = filtered[
        ~filtered.index.isin(already_recommended)
    ]

    if filtered.empty:
        print(Fore.RED + "\nNo movies match your preferences.")
        return None

    # Build a personalized search query
    query_parts = []

    if genre:
        query_parts.append(genre)

    if mood:
        query_parts.append(MOODS[mood])

    if not query_parts:
        query_parts.append(
            "interesting memorable entertaining cinematic story"
        )

    query = " ".join(query_parts)

    # TF-IDF converts movie descriptions into numerical vectors
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=10000,
        ngram_range=(1, 2)
    )

    # Fit on all movie descriptions
    all_vectors = vectorizer.fit_transform(
        movies["combined_features"]
    )

    # Convert user preferences into a vector
    user_vector = vectorizer.transform([query])

    # Calculate cosine similarity
    similarity_scores = cosine_similarity(
        user_vector,
        all_vectors
    ).flatten()

    # Attach scores to filtered movies
    filtered = filtered.copy()

    filtered["similarity"] = similarity_scores[
        filtered.index
    ]

    # Highest similarity comes first
    filtered = filtered.sort_values(
        by=["similarity", "imdb_rating"],
        ascending=[False, False]
    )

    return filtered.iloc[0]


# ==========================================
#          RANDOM RECOMMENDATION
# ==========================================

def random_recommendation(
    movies,
    genre,
    minimum_rating,
    already_recommended
):

    filtered = filter_movies(
        movies,
        genre,
        minimum_rating
    )

    filtered = filtered[
        ~filtered.index.isin(already_recommended)
    ]

    if filtered.empty:
        print(Fore.RED + "\nNo more movies match your preferences.")
        return None

    return filtered.sample(n=1).iloc[0]


# ==========================================
#           DISPLAY MOVIE DETAILS
# ==========================================

def display_movie(movie, method):

    print("\n")

    print_line()
    print(Fore.YELLOW + Style.BRIGHT + "YOUR MOVIE RECOMMENDATION")
    print_line()

    print(
        Fore.CYAN + Style.BRIGHT +
        f"\nTITLE: {movie['series_title']}"
    )

    print(
        Fore.WHITE +
        f"\nGENRE: {movie['genre']}"
    )

    print(
        Fore.WHITE +
        f"\nIMDb RATING: {movie['imdb_rating']}/10"
    )

    print(
        Fore.WHITE +
        "\nOVERVIEW:"
    )

    print(
        Fore.LIGHTWHITE_EX +
        f"{movie['overview']}"
    )

    # Sentiment analysis
    sentiment, color = analyze_sentiment(
        movie["overview"]
    )

    print(
        Fore.WHITE +
        "\nOVERVIEW SENTIMENT: " +
        color + Style.BRIGHT + sentiment
    )

    if method == "AI":

        print(
            Fore.MAGENTA +
            f"\nAI SIMILARITY: {movie['similarity'] * 100:.2f}%"
        )

    elif method == "RANDOM":

        print(
            Fore.MAGENTA +
            "\nSELECTION METHOD: Random"
        )

    print_line()


# ==========================================
#              MAIN PROGRAM
# ==========================================

def main():

    welcome_screen()

    # Load dataset
    movies = load_movies()

    if movies is None:
        return

    print(
        Fore.GREEN +
        f"Successfully loaded {len(movies)} movies!"
    )

    # Ask for user name
    name = input(
        Fore.YELLOW + "\nWhat is your name? "
    ).strip()

    if not name:
        name = "Movie Fan"

    print(
        Fore.GREEN +
        f"\nHello, {name}! Let's find you a movie."
    )

    # Store previously recommended movies
    already_recommended = set()

    while True:

        print("\n")
        print_line()
        print(Fore.YELLOW + Style.BRIGHT + "MAIN MENU")
        print_line()

        print(Fore.CYAN + "1. AI Movie Recommendation")
        print(Fore.CYAN + "2. Random Movie Recommendation")
        print(Fore.CYAN + "3. Exit")

        choice = input(
            Fore.GREEN + "\nChoose an option: "
        ).strip()

        if choice == "3":

            print(
                Fore.YELLOW +
                f"\nThanks for using the Movie Recommender, {name}!"
            )

            print(Fore.GREEN + "Have a great day!")
            break

        elif choice not in ["1", "2"]:

            print(Fore.RED + "\nInvalid option. Choose 1, 2, or 3.")
            continue

        # Get preferences
        genre = choose_genre(movies)

        mood = None

        if choice == "1":
            mood = choose_mood()

        minimum_rating = choose_rating()

        # Generate recommendation
        if choice == "1":

            movie = ai_recommendation(
                movies,
                genre,
                mood,
                minimum_rating,
                already_recommended
            )

            method = "AI"

        else:

            movie = random_recommendation(
                movies,
                genre,
                minimum_rating,
                already_recommended
            )

            method = "RANDOM"

        if movie is None:
            continue

        # Remember recommended movie
        already_recommended.add(movie.name)

        # Display recommendation
        display_movie(movie, method)

        # Repeat option
        while True:

            again = input(
                Fore.YELLOW +
                "\nWould you like another recommendation? (yes/no): "
            ).strip().lower()

            if again in ["yes", "y"]:
                break

            elif again in ["no", "n"]:
                print(
                    Fore.GREEN +
                    f"\nEnjoy your movie, {name}! Goodbye!"
                )
                return

            else:
                print(Fore.RED + "Please enter yes or no.")


# ==========================================
#             START APPLICATION
# ==========================================

if __name__ == "__main__":
    main()
