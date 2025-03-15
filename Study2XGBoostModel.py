import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------- MFT Probability Parsing -------------------- #
def parse_mft_probability(cell):
    """
    Convert strings like 'authority (0.95)' or 'authority(95)' into a float (e.g. 0.95).
    If parsing fails or the string starts with 'not ', return NaN.
    """
    if isinstance(cell, str):
        if cell.lower().startswith("not "):
            return np.nan
        match = re.search(r'\((.*?)\)', cell)
        if match:
            val_str = match.group(1).strip()
            if '.' in val_str:
                try:
                    return float(val_str)
                except ValueError:
                    return np.nan
            else:
                try:
                    return float(val_str) / 100.0
                except ValueError:
                    return np.nan
    return np.nan

# -------------------- File Paths -------------------- #
politicians_path = r"C:\Users\feeha\iCloudDrive\PhD Research\IRA\Study 2 and 3 Final Data\PoliticansEmotionMFTPartisanCleanedPartisanWordCountKeysFixed.csv"
engaged_users_path = r"C:\Users\feeha\iCloudDrive\PhD Research\IRA\Study 2 and 3 Final Data\politicalResultsFullAllFunctionsKeysFixed.csv"
random_users_path   = r"C:\Users\feeha\iCloudDrive\PhD Research\IRA\Study 2 and 3 Final Data\randomusersResultsFullAllFunctionsKeysFixed.csv"
russian_path        = r"C:\Users\feeha\iCloudDrive\PhD Research\IRA\Study 2 and 3 Final Data\3millionTweetsCleanedPartisanWordsIVKeysFixed.csv"

# -------------------- Step 1: Load & Combine Labeled Datasets -------------------- #
# Load labeled datasets
politicians = pd.read_csv(politicians_path)
engaged_users = pd.read_csv(engaged_users_path)
random_users = pd.read_csv(random_users_path)

# Add author type labels
politicians['author_type'] = 'politician'
engaged_users['author_type'] = 'engaged_user'
random_users['author_type'] = 'random_user'

# Combine the three labeled datasets
labeled_df = pd.concat([politicians, engaged_users, random_users], ignore_index=True)

# -------------------- Parse MFT Columns into Floats -------------------- #
mft_columns = [
    'Loyaltyclassification_result',
    'Sanctityclassification_result',
    'Careclassification_result',
    'Fairnessclassification_result',
    'Authorityclassification_result'
]

for col in mft_columns:
    if col in labeled_df.columns:
        labeled_df[col] = labeled_df[col].apply(parse_mft_probability)

# -------------------- Step 2: Feature Engineering -------------------- #
# Define the features.
# (Assumes that the emotion keys have been expanded previously into separate numeric columns)
features = [
    'fear', 'anger', 'trust', 'surprise', 'sadness', 'disgust', 'joy',
    'Loyaltyclassification_result', 'Sanctityclassification_result',
    'Careclassification_result', 'Fairnessclassification_result',
    'Authorityclassification_result', 'partisan_word_count', 'follower_count'
]

X = labeled_df[features]

# Map string labels to numeric values for classification
label_mapping = {'politician': 0, 'engaged_user': 1, 'random_user': 2}
labeled_df['author_type_numeric'] = labeled_df['author_type'].map(label_mapping)
y = labeled_df['author_type_numeric']

# -------------------- Step 3: Train-Test Split -------------------- #
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------- Step 4: Train XGBoost Model with GPU -------------------- #
# Updated GPU parameters: 
# - Use the standard histogram algorithm (tree_method='hist')
# - Specify device='cuda' for GPU acceleration
model = XGBClassifier(
    objective='multi:softprob',
    eval_metric='mlogloss',
    tree_method='hist',  # Use histogram-based algorithm
    device='cuda',       # Use GPU for training
    num_class=3,         # Only three classes for labeled data
    random_state=42
)

print("Training the XGBoost model with GPU acceleration on labeled data...")
model.fit(X_train, y_train)

# -------------------- Step 5: Model Evaluation on Labeled Data -------------------- #
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)

print("Classification Report (Labeled Data):")
print(classification_report(y_test, y_pred, target_names=list(label_mapping.keys())))

conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=list(label_mapping.keys()),
            yticklabels=list(label_mapping.keys()))
plt.title("Confusion Matrix (Labeled Data)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

feature_importances = pd.DataFrame({
    'Feature': features,
    'Importance': model.feature_importances_
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importances)
plt.title("Feature Importance")
plt.show()

# -------------------- Step 6: Predict on Russian IRA Messages -------------------- #
# Load the Russian dataset (unlabeled)
russian_df = pd.read_csv(russian_path)

# Parse MFT columns in the Russian data
for col in mft_columns:
    if col in russian_df.columns:
        russian_df[col] = russian_df[col].apply(parse_mft_probability)

# (Assumes that the same feature engineering was applied to expand emotion keys in the Russian data)
X_russian = russian_df[features]

# Predict author type for Russian messages using the trained model
russian_predictions = model.predict(X_russian)
russian_pred_proba = model.predict_proba(X_russian)

# Map numeric predictions back to string labels
inverse_label_map = {v: k for k, v in label_mapping.items()}
russian_df['predicted_author_type'] = [inverse_label_map[int(pred)] for pred in russian_predictions]

# Add prediction confidence (maximum probability across classes)
russian_df['prediction_confidence'] = russian_pred_proba.max(axis=1)

# Specify the full output path
russian_output_path = r"C:\Users\feeha\iCloudDrive\PhD Research\IRA\Study 2 and 3 Final Data\predicted_russian_messages.csv"

# Save the predictions to the specified path
russian_df.to_csv(russian_output_path, index=False)
print(f"Predicted labels for Russian messages saved to '{russian_output_path}'.")

# (Optional) Print counts of predicted categories
print("\nPredicted distribution for Russian messages:")
print(russian_df['predicted_author_type'].value_counts())

