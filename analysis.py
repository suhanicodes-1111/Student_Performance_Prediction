import pandas as pd

data = pd.read_csv("data/student-mat.csv", sep=";")

print(data.head())
print(data.info())
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(8,5))

sns.histplot(data["G3"], bins=20)

plt.title("Distribution of Final Grades")

plt.xlabel("Final Grade (G3)")
plt.ylabel("Number of Students")

plt.show()
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Select features
features = data[["studytime","failures","absences","G1","G2"]]

# Target variable
target = data["G3"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    features, target, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

print("Predictions:", predictions[:10])
from sklearn.metrics import mean_absolute_error

error = mean_absolute_error(y_test, predictions)

print("Mean Absolute Error:", error)
import matplotlib.pyplot as plt

plt.scatter(y_test, predictions)

plt.xlabel("Actual Grades")
plt.ylabel("Predicted Grades")

plt.title("Actual vs Predicted Student Grades")

plt.show()