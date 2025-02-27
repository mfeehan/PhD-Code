import pandas as pd
import statsmodels.api as sm
import numpy as np

# Load the CSV file
df = pd.read_csv("/Users/mathewfeehan/Library/Mobile Documents/com~apple~CloudDocs/PhD Research/IRA/TestOutputs/Study 3 Final Data/3millionTweetsCleanedPartisanWordsIV.csv")

# Print column names
for c in df.columns:
    print(c)

# Define MFT independent variables
mft_ivs = [
    "Loyaltyclassification_result",
    "Sanctityclassification_result",
    "Careclassification_result",
    "Fairnessclassification_result",
    "Authorityclassification_result",
]

# Process the 'affect' column
emotion_df = pd.DataFrame([eval(v) for v in df["affect"]])

# Fill missing values in the DataFrame
emotion_df = emotion_df.fillna(0)

# Drop specific column
emotion_clean_df = emotion_df.drop(columns=["anticip", "anticipation"])

# Get relevant index based on sum condition
rel_index = emotion_clean_df[emotion_clean_df.sum(axis=1) > 0].index

# Update the relevant index by dropping NaN retweet_count
rel_index = df.loc[rel_index].dropna(subset=["retweet_count"]).index

# Length of relevant index
print(len(rel_index))

# Function to run OLS and GLM regressions
def run_regressions(X, y_ols, y_glm, description):
    # Run OLS regression
    ols_results = sm.OLS(y_ols, X).fit()
    print(f"OLS Regression Results ({description}):")
    print(ols_results.summary())

    # Run GLM regression
    glm_results = sm.GLM(
        y_glm,
        X,
        family=sm.families.NegativeBinomial(alpha=2)
    ).fit(cov_type="HC2")
    print(f"GLM Regression Results ({description}):")
    print(glm_results.summary())

# Define dependent variables
y_ols = np.log2(df.loc[rel_index]["retweet_count"].copy() + 1)
y_glm = df.loc[rel_index]["like_count"].copy()

# 1. Emotion variables only
X_emotion = sm.add_constant(emotion_clean_df.loc[rel_index].drop(columns=["negative", "positive"]))
X_emotion["follower_count"] = np.log2(df.loc[rel_index]["follower_count"] + 1)  # Add follower_count
run_regressions(X_emotion, y_ols, y_glm, "Emotion Variables Only")

# 2. Emotion + MFT classification
X_emotion_mft = X_emotion.copy()
for mft_v in mft_ivs:
    X_emotion_mft[mft_v] = df.loc[rel_index][mft_v].apply(
        lambda s: float(s.partition("(")[-1].partition(")")[0]) if type(s) == str else 0
    )
X_emotion_mft["follower_count"] = np.log2(df.loc[rel_index]["follower_count"] + 1)  # Add follower_count
run_regressions(X_emotion_mft, y_ols, y_glm, "Emotion + MFT Classification Variables")

# 3. Emotion + MFT classification + Partisan Word Count
X_emotion_mft_partisan = X_emotion_mft.copy()
X_emotion_mft_partisan["partisan_word_count"] = df.loc[rel_index]["partisan_word_count"]
run_regressions(X_emotion_mft_partisan, y_ols, y_glm, "Emotion + MFT + Partisan Word Count")
