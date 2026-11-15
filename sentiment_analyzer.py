import nltk
from nltk.corpus import movie_reviews
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

nltk.download('movie_reviews', quiet=True)

#Loads the data into 2 lists. the text of the review and if the review was positive or negative (noted by a 1 or a 0)
def load_data():
    texts = []
    labels = []
    for category in movie_reviews.categories():
        if category == "pos":
            label = 1
        else:
            label = 0
        for  fileid in movie_reviews.fileids(category):
            texts.append(preprocess_text(movie_reviews.raw(fileid)))
            labels.append(label)
    total = len(labels)
    pos_count = sum(labels)
    neg_count = total - pos_count
    print(f"total: {total} | Positive {pos_count} | Negative {neg_count}" )
    return texts, labels

#function that cleans text. Example being "Great!" to "great"
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text
    
#Train and Evaluate function that unloads and process the data. 
def train_and_evaluate():
    #loads the data into the 2 variable, texts and labels
    texts, labels = load_data()
    #splits the data based off of a 80/20 split
    X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42, stratify=labels)
    #creates a seperate vectorizer that we can use off of the Source Vectorizer function
    vectorizer = CountVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vec, y_train)
    y_pred = model.predict(X_test_vec)

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print("\n--- Model Evaluation ---")
    print(f"Test Accuracy: {acc:.2%}")
    print("\nConfusion Matrix:")
    print("              Pred Neg   Pred Pos")
    print(f"Actual Neg      {cm[0][0]:<10} {cm[0][1]:<10}")
    print(f"Actual Pos      {cm[1][0]:<10} {cm[1][1]:<10}")
    print("\nWhat the model sees:")
    print("With a bag-of-words representation, the model only sees which words appear and how often, with zero record of word order.")
    return vectorizer, model

# Runs the load_data function when the program is started. 
if __name__ == "__main__":
    train_and_evaluate()
