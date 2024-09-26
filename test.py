# # import pandas as pd
# # import pickle

# # # Load the data from the CSV file
# # data = pd.read_csv('heart_cleveland_upload.csv')

# # # Load the model from the .pkl file
# # with open('heart-disease-prediction-knn-model.pkl', 'rb') as file:
# #     loaded_model = pickle.load(file)

# # # Use the loaded model to make predictions on the data
# # predictions = loaded_model.predict(data)

# # # Print the predictions
# # print(predictions)
# import pandas as pd
# import pickle

# # Load the data from the CSV file
# data = pd.read_csv('heart_cleveland_upload.csv')

# # Assuming the last column is the actual result
# feature_columns = data.columns[:-1]
# actual_result_column = data.columns[-1]

# # Separate features and actual results
# X = data[feature_columns]
# y_actual = data[actual_result_column]

# # Load the model from the .pkl file
# with open('heart-disease-prediction-knn-model.pkl', 'rb') as file:
#     loaded_model = pickle.load(file)

# # Use the loaded model to make predictions on the data
# y_predicted = loaded_model.predict(X)

# # Combine features, actual results, and predicted results into a single DataFrame
# result_df = X.copy()
# result_df['Actual'] = y_actual
# result_df['Predicted'] = y_predicted

# # Display 30 random samples from the DataFrame
# random_samples = result_df.sample(n=30, random_state=1)
# print(random_samples)

# # If you want to display it as a table in a more readable format, you can use the following:
# import prettytable as pt

# table = pt.PrettyTable()
# table.field_names = list(result_df.columns)

# for _, row in random_samples.iterrows():
#     table.add_row(row)

# print(table)


import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

# Load the data from the CSV file
data = pd.read_csv('heart_cleveland_upload.csv')

# Assuming the last column is the actual result
feature_columns = data.columns[:-1]
actual_result_column = data.columns[-1]

# Separate features and actual results
X = data[feature_columns]
y_actual = data[actual_result_column]

# Load the model from the .pkl file
with open('heart-disease-prediction-knn-model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

# Apply scaling using the same scaler as used during training
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Use the loaded model to make predictions on the data
y_predicted = loaded_model.predict(X_scaled)

# Combine features, actual results, and predicted results into a single DataFrame
result_df = X.copy()
result_df['Actual'] = y_actual
result_df['Predicted'] = y_predicted


# Display the entire DataFrame
print(result_df.to_string(index=False))
