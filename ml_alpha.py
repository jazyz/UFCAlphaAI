import csv
import pandas as pd
import sys
import lightgbm as lgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
import numpy as np

# Step 1: Read the data
df = pd.read_csv("data\detailed_fights.csv")
# Step 2: Preprocess the data
# Assuming 'Result' is the target variable and the rest are features
label_encoder = LabelEncoder()
df["Result"] = label_encoder.fit_transform(df["Result"])

selected_columns = [
    "Result",
    # "Red Fighter",
    # "Blue Fighter",
    # "Title",
    "Red dob",
    "Blue dob",
    "Red totalfights",
    "Blue totalfights",
    "Red elo",
    "Blue elo",
    "Red losestreak",
    "Blue losestreak",
    "Red winstreak",
    "Blue winstreak",
    "Red titlewins",
    "Blue titlewins",
    "Red oppelo",
    "Blue oppelo",
    "Red wins",
    "Blue wins",
    "Red losses",
    "Blue losses",
    "Red KD",
    "Blue KD",
    "Red KD differential",
    "Blue KD differential",
    "Red Sig. str.",
    "Blue Sig. str.",
    "Red Sig. str. differential",
    "Blue Sig. str. differential",
    "Red Total str.",
    "Blue Total str.",
    "Red Total str. differential",
    "Blue Total str. differential",
    "Red Td",
    "Blue Td",
    "Red Td differential",
    "Blue Td differential",
    "Red Sub. att",
    "Blue Sub. att",
    "Red Sub. att differential",
    "Blue Sub. att differential",
    "Red Rev.",
    "Blue Rev.",
    "Red Rev. differential",
    "Blue Rev. differential",
    "Red Ctrl",
    "Blue Ctrl",
    "Red Ctrl differential",
    "Blue Ctrl differential",
    "Red Head",
    "Blue Head",
    "Red Head differential",
    "Blue Head differential",
    "Red Body",
    "Blue Body",
    "Red Body differential",
    "Blue Body differential",
    "Red Leg",
    "Blue Leg",
    "Red Leg differential",
    "Blue Leg differential",
    "Red Distance",
    "Blue Distance",
    "Red Distance differential",
    "Blue Distance differential",
    "Red Clinch",
    "Blue Clinch",
    "Red Clinch differential",
    "Blue Clinch differential",
    "Red Ground",
    "Blue Ground",
    "Red Ground differential",
    "Blue Ground differential",
    "Red Sig. str.%",
    "Blue Sig. str.%",
    "Red Sig. str.% differential",
    "Blue Sig. str.% differential",
    "Red Sig. str.% defense",
    "Blue Sig. str.% defense",
    "Red Total str.%",
    "Blue Total str.%",
    "Red Total str.% differential",
    "Blue Total str.% differential",
    "Red Total str.% defense",
    "Blue Total str.% defense",
    "Red Td%",
    "Blue Td%",
    "Red Td% differential",
    "Blue Td% differential",
    "Red Td% defense",
    "Blue Td% defense",
    "Red Head%",
    "Blue Head%",
    "Red Head% differential",
    "Blue Head% differential",
    "Red Head% defense",
    "Blue Head% defense",
    "Red Body%",
    "Blue Body%",
    "Red Body% differential",
    "Blue Body% differential",
    "Red Body% defense",
    "Blue Body% defense",
    "Red Leg%",
    "Blue Leg%",
    "Red Leg% differential",
    "Blue Leg% differential",
    "Red Leg% defense",
    "Blue Leg% defense",
    "Red Distance%",
    "Blue Distance%",
    "Red Distance% differential",
    "Blue Distance% differential",
    "Red Distance% defense",
    "Blue Distance% defense",
    "Red Clinch%",
    "Blue Clinch%",
    "Red Clinch% differential",
    "Blue Clinch% differential",
    "Red Clinch% defense",
    "Blue Clinch% defense",
    "Red Ground%",
    "Blue Ground%",
    "Red Ground% differential",
    "Blue Ground% differential",
    "Red Ground% defense",
    "Blue Ground% defense",
    "dob oppdiff",
    "totalfights oppdiff",
    "elo oppdiff",
    "losestreak oppdiff",
    "winstreak oppdiff",
    "titlewins oppdiff",
    "oppelo oppdiff",
    "wins oppdiff",
    "losses oppdiff",
    "KD oppdiff",
    "KD differential oppdiff",
    "Sig. str. oppdiff",
    "Sig. str. differential oppdiff",
    "Total str. oppdiff",
    "Total str. differential oppdiff",
    "Td oppdiff",
    "Td differential oppdiff",
    "Sub. att oppdiff",
    "Sub. att differential oppdiff",
    "Rev. oppdiff",
    "Rev. differential oppdiff",
    "Ctrl oppdiff",
    "Ctrl differential oppdiff",
    "Head oppdiff",
    "Head differential oppdiff",
    "Body oppdiff",
    "Body differential oppdiff",
    "Leg oppdiff",
    "Leg differential oppdiff",
    "Distance oppdiff",
    "Distance differential oppdiff",
    "Clinch oppdiff",
    "Clinch differential oppdiff",
    "Ground oppdiff",
    "Ground differential oppdiff",
    "Sig. str.% oppdiff",
    "Sig. str.% differential oppdiff",
    "Sig. str.% defense oppdiff",
    "Total str.% oppdiff",
    "Total str.% differential oppdiff",
    "Total str.% defense oppdiff",
    "Td% oppdiff",
    "Td% differential oppdiff",
    "Td% defense oppdiff",
    "Head% oppdiff",
    "Head% differential oppdiff",
    "Head% defense oppdiff",
    "Body% oppdiff",
    "Body% differential oppdiff",
    "Body% defense oppdiff",
    "Leg% oppdiff",
    "Leg% differential oppdiff",
    "Leg% defense oppdiff",
    "Distance% oppdiff",
    "Distance% differential oppdiff",
    "Distance% defense oppdiff",
    "Clinch% oppdiff",
    "Clinch% differential oppdiff",
    "Clinch% defense oppdiff",
    "Ground% oppdiff",
    "Ground% differential oppdiff",
    "Ground% defense oppdiff",
]


df = df[selected_columns]
X = df.drop(["Result"], axis=1)
y = df["Result"]

# Convert categorical variables if any
# X = pd.get_dummies(X)  # This line is optional and depends on your data

# Manual split based on percentage
split_index = int(len(df) * 0.8)
last_index = int(len(df) * 1)
X_train, X_test = X[:split_index], X[split_index:last_index]
y_train, y_test = y[:split_index], y[split_index:last_index]

seed = 42

# Determine the new start index for the training data to skip the first 20%
prune_index = int(len(X_train) * 0.1)


X_train = X_train[prune_index:]
y_train = y_train[prune_index:]


# Step 1: Duplicate the training data
X_train_swapped = X_train.copy()
y_train_swapped = y_train.copy()

# Step 2: Rename the columns to swap 'Red' with 'Blue'
swap_columns = {}
for column in X_train.columns:
    if "Red" in column:
        swap_columns[column] = column.replace("Red", "Blue")
    elif "Blue" in column:
        swap_columns[column] = column.replace("Blue", "Red")

# Rename the columns in the copied DataFrame
X_train_swapped.rename(columns=swap_columns, inplace=True)

# Inverse the target variable for the swapped data
# Assuming 'win', 'loss', and 'draw' are the possible values
y_train_swapped = y_train_swapped.apply(lambda x: 2 if x == 1 else (1 if x == 2 else 0))

# Step 3: Concatenate the original and the modified copy to form the extended training set
X_train_extended = pd.concat([X_train, X_train_swapped], ignore_index=True)
y_train_extended = pd.concat([y_train, y_train_swapped], ignore_index=True)


# model.fit(X_train, y_train)
model = lgb.LGBMClassifier(random_state=seed)
model.fit(X_train_extended, y_train_extended)
# Make predictions and evaluate the model
y_pred = model.predict(X_test)
predicted_probabilities = model.predict_proba(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# Get the fighter names and actual results for the test set
df_with_details = pd.read_csv("data\detailed_fights.csv")[
    ["Red Fighter", "Blue Fighter", "Result"]
]
df_with_details = df_with_details.iloc[split_index:]  # Align with the test data split
df_with_details.reset_index(drop=True, inplace=True)
df_with_details["Result"] = label_encoder.fit_transform(df_with_details["Result"])

# Convert the predicted and actual results back to the original labels if necessary.
predicted_labels = label_encoder.inverse_transform(y_pred)
actual_labels = label_encoder.inverse_transform(df_with_details["Result"])

with open("data\predicted_results.csv", mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(
        [
            "Red Fighter",
            "Blue Fighter",
            "Predicted Result",
            "Probability",
            "Actual Result",
        ]
    )
    for i in range(len(predicted_labels)):
        max_probability = max(predicted_probabilities[i])

        writer.writerow(
            [
                df_with_details["Red Fighter"].iloc[i],
                df_with_details["Blue Fighter"].iloc[i],
                predicted_labels[i],
                max_probability,  # Formatting as a percentage
                actual_labels[i],
            ]
        )

feature_importances = model.feature_importances_

feature_importance_df = pd.DataFrame(
    {"Feature": X_train.columns, "Importance": feature_importances}
)

feature_importance_df = feature_importance_df.sort_values("Importance", ascending=False)

plt.figure(figsize=(10, 6))
plt.barh(feature_importance_df["Feature"], feature_importance_df["Importance"])
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Feature Importance")
plt.show()

print("Top 25 Important Features:")
print(feature_importance_df.head(10))


################### HYPERPARAMETERS
# def lgb_cv(num_leaves, learning_rate, min_data_in_leaf, reg_alpha):
#     """LGBM Cross Validation Score.
#     This function will be passed to Bayesian Optimizer.
#     """
#     params = {
#         'num_leaves': int(round(num_leaves)),
#         'learning_rate': learning_rate,
#         'min_data_in_leaf': int(round(min_data_in_leaf)),
#         'reg_alpha': reg_alpha,
#         'random_state': seed  # ensure reproducibility
#     }

#     # Initialize and cross-validate the model
#     model = lgb.LGBMClassifier(**params)
#     cv_score = cross_val_score(model, X_train_extended, y_train_extended, cv=5, scoring='accuracy').mean()
#     return cv_score

# # Define bounds of hyperparameters for Bayesian Optimization
# pbounds = {
#     'num_leaves': (20, 60),
#     'learning_rate': (0.001, 0.1),
#     'min_data_in_leaf': (20, 200),
#     'reg_alpha': (0, 1)
# }

# # Initialize Bayesian Optimization
# optimizer = BayesianOptimization(
#     f=lgb_cv,
#     pbounds=pbounds,
#     random_state=42,
# )

# # Optimize
# optimizer.maximize(
#     init_points=2,  # number of initializing random points
#     n_iter=10,  # number of iterations for optimization
# )

# # Extract the best parameters
# best_params = optimizer.max['params']

# # Convert the 'num_leaves' and 'min_data_in_leaf' to int since they must be integers
# best_params['num_leaves'] = int(round(best_params['num_leaves']))
# best_params['min_data_in_leaf'] = int(round(best_params['min_data_in_leaf']))

# print("Best parameters:", best_params)

# # Now use the best parameters to fit the model
# model = lgb.LGBMClassifier(
#     **best_params,
#     random_state=seed
# )
# Fit the model
# best_params = {
#     "learning_rate": 0.1,
#     "min_data_in_leaf": 30,
#     "num_leaves": 31,
#     "reg_alpha": 0.1,
# }

# Initialize and train the model
# model = lgb.LGBMClassifier(
#     random_state=seed,
#     learning_rate=best_params["learning_rate"],
#     min_data_in_leaf=best_params["min_data_in_leaf"],
#     num_leaves=best_params["num_leaves"],
#     reg_alpha=best_params["reg_alpha"],
# )
# model = lgb.LGBMClassifier(
#     random_state=42,
#     num_leaves=31,
#     learning_rate=0.1,
#     min_data_in_leaf=30,
#     reg_alpha=0.1,
#     max_bin=200,
#     max_depth=15
# )
