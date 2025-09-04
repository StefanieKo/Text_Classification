
#### Sentiment Analysis
#### Kann ein Decision-Tree-Modell die Stimmung (Sentiment) eines Textes aus den Wortmustern zuverlässig vorhersagen?####
#### 1. Bag-of-Words Modell, 2. Decision-Tree-Classifier, 3. Modellqualität(Accuracy, Precision, Recall)


import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score, recall_score, classification_report
import numpy as np

# CSV-Datei von GitHub laden
url = "https://raw.githubusercontent.com/cornelius31415/DATA-SCIENCE/main/Sentiment%20Analysis/sentiment_analysis.csv"
sentiment_data = pd.read_csv(url)

# Textspalte als Liste
sentiment_sentences = sentiment_data['text'].astype(str).tolist()

# Bag-of-Words Modell erzeugen
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(sentiment_sentences)

# In DataFrame umwandeln
bow_df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())

# Zielvariable
y = sentiment_data["sentiment"]

# Aufteilen in Trainings- und Testdaten
X_train, X_test, y_train, y_test = train_test_split(bow_df, y, test_size=0.3, random_state=0)

# Modell erstellen und trainieren
tree = DecisionTreeClassifier(random_state=1)
tree.fit(X_train, y_train)

# Vorhersage auf Testdaten
y_pred = tree.predict(X_test)

# Accuracy: richtige Vorhersagen
accuracy = tree.score(X_test, y_test)
print(f"Accuracy: {accuracy:.2f}")

# Precision & Recall
## Precision: Wie genau sind die Vorhersagen? Precision = True Positives / (True Positives + False Positives)

precision = precision_score(y_test, y_pred, average='macro')

## Recall: Wie viele der tatsächlich positiven Fälle wurden erkann? Recall = True Positives / (True Positives + False Negatives)

recall = recall_score(y_test, y_pred, average='macro')

print(f"Precision (macro): {precision:.2f}")
print(f"Recall (macro): {recall:.2f}")

# Optional: Vollständiger Bericht
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
