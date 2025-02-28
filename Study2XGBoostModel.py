import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load and Combine Datasets (Without Russian Labels)
politicians_path = 'path_to_politicians.csv'
engaged_users_path = 'path_to_engaged_users.csv'
random_users_path = 'path_to_random_users.csv'
russian_path = 'path_to_russian.csv'

# Load datasets
politicians = pd.read_csv(politicians_path)
engaged_users = pd.read_csv(engaged_users_path)
random_users = pd.read_csv(random_users_path)
russian = pd.read_csv(russian_path)

# Add labels for training data
politicians['author_type'] = 'politician'
engaged_users['author_type'] = 'engaged_user'
random_users['author_type'] = 'random_user'

# Combine labeled datasets
combined_df = pd.concat([politicians, engaged_users, random_users], ignore_index=True)

# Step 2: Feature Engineering
# Select relevant features
features = [
    'fear', 'anger', 'trust', 'surprise', 'sadness', 'disgust', 'joy',
    'Loyaltyclassification_result', 'Sanctityclassification_result',
    'Careclassification_result', 'Fairnessclassification_result',
    'Authorityclassification_result', 'partisan_word_count', 'follower_count'
]
X = combined_df[features]

# Convert labels to numeric
label_mapping = {'politician': 0, 'engaged_user': 1, 'random_user': 2}
combined_df['author_type_numeric'] = combined_df['author_type'].map(label_mapping)
y = combined_df['author_type_numeric']

# Step 3: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Step 4: Train the XGBoost Model
model = XGBClassifier(
    objective='multi:softprob', 
    eval_metric='mlogloss', 
    use_label_encoder=False,
    num_class=3,
    random_state=42
)

print("Training the XGBoost model...")
model.fit(X_train, y_train)

# Step 5: Model Evaluation
y_pred = model.predict(X_test)

# Classification report
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=label_mapping.keys()))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=label_mapping.keys(), yticklabels=label_mapping.keys())
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Feature Importance
feature_importances = pd.DataFrame({
    'Feature': features,
    'Importance': model.feature_importances_
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importances)
plt.title("Feature Importance")
plt.show()

# Step 6: Predict Russian Dataset
russian_features = russian[features]
russian_predictions = model.predict(russian_features)

# Map predictions to labels
russian['predicted_author_type'] = [list(label_mapping.keys())[int(pred)] for pred in russian_predictions]

# Calculate the percentage of tweets for each predicted label
label_counts = russian['predicted_author_type'].value_counts()
label_percentages = (label_counts / len(russian)) * 100

# Print percentages
print("Percentage of tweets predicted for each label:")
print(label_percentages)

# Visualize the percentages in a pie chart
plt.figure(figsize=(8, 6))
label_percentages.plot(kind='pie', autopct='%1.1f%%', startangle=140, colors=['lightblue', 'lightgreen', 'salmon'])
plt.title("Distribution of Predicted Author Types in Russian Dataset")
plt.ylabel("")  # Remove the y-axis label
plt.show()

# Save predictions
russian.to_csv('predicted_russian_tweets.csv', index=False)
print("Predictions saved to 'predicted_russian_tweets.csv'")
