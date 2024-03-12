import pandas as pd
import json


data = json.load(open('article_data.json'))

# Assuming `data` is your loaded JSON data
df = pd.DataFrame(data["data"])

# Convert `created_at` to datetime and extract year, month, etc.
df['created_at'] = pd.to_datetime(df['created'], errors='coerce')
df['year'] = df['created_at'].dt.year
df['month'] = df['created_at'].dt.month
# Add more temporal features as needed

# Drop the original `created_at` column if no longer needed
df.drop(['created_at'], axis=1, inplace=True)



from sklearn.feature_extraction.text import TfidfVectorizer

# Vectorize `title` + `text`
vectorizer = TfidfVectorizer(max_features=10000)  # Adjust as necessary
text_features = vectorizer.fit_transform(df['title'] + ' ' + df['text']).toarray()



import numpy as np

# Assuming `year` and `month` are the only additional features you've prepared
additional_features = df[['year', 'month']].values  # Convert to numpy array if not already

# Combine additional features with text features
X = np.hstack([text_features, additional_features])
y = df['label'].values  # Replace 'label' with your actual target variable column name



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


import xgboost as xgb

# Convert the dataset into an optimized data structure DMatrix
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# Define parameters
params = {
    'max_depth': 6,  # the maximum depth of each tree
    'eta': 0.3,  # the training step for each iteration
    'objective': 'multi:softprob',  # error evaluation for multiclass training
    'num_class': 3  # the number of classes that exist in this dataset
}
epochs = 10  # The number of training iterations

# Train the model
model = xgb.train(params, dtrain, epochs)

# Predictions
predictions = model.predict(dtest)


from sklearn.metrics import accuracy_score

# Convert probabilities to predicted class labels
predictions = np.asarray([np.argmax(line) for line in predictions])

# Calculate accuracy
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy}")
