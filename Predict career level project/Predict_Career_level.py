import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble._forest import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.compose import ColumnTransformer

def transform_location(column):
    pattern = re.findall('[A-Z]{2}$', column)
    if pattern:
        return ''.join(pattern)
    else:
        return column


data = pd.read_excel('data.ods', engine='odf', dtype=str)
data = data.dropna(axis=0)
X = data.drop('career_level', axis=1)
X['location'] = X['location'].apply(transform_location)
y = data['career_level']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=23, stratify=data['career_level'])


# Preprocessing each features
preprocessor = ColumnTransformer(transformers=[
    ("title_feature", TfidfVectorizer(stop_words="english", ngram_range=(1, 1)), "title"),
    ("location_feature", OneHotEncoder(handle_unknown='ignore'), ["location"]),
    ("description_feature", TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=0.023, max_df=0.96), "description"),
    ("function_feature", OneHotEncoder(handle_unknown='ignore'), ["function"]),
    ("industry_feature", TfidfVectorizer(stop_words="english", ngram_range=(1, 1)), "industry")
])

#
model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ("model", DecisionTreeClassifier())
])

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))