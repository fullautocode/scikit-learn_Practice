import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load the dataset from the URL
url = "https://raw.githubusercontent.com/gheniabla/datasets/master/score.csv"
data = pd.read_csv(url)

# Split the data into training and testing sets (75% train, 25% test)
train_data, test_data = train_test_split(data, test_size=0.25, random_state=42)

# Extracting the feature (Hours) and the label (Scores)
X_train = train_data[['Hours']]
y_train = train_data['Scores']

# Initialize and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Print the model intercept
print("Intercept:", model.intercept_)

# Print the model coefficient (slope)
print("Slope:", model.coef_[0])

# Prediction for a student who studied 7.56 hours
predicted_score = model.predict([[7.56]])
print("Predicted Score for 7.56 hours of study:", predicted_score[0])