import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

# Load & Prep
df = pd.read_csv('student_performance_1000_rows.csv')
X = df[['Study_Hours', 'Attendance', 'Previous_Score', 'Assignments_Completed']]
y = df['Final_Score']

# Train-Test Split (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Building & Training
model = LinearRegression().fit(X_train, y_train)

# Quick Evaluation
print(f"Model accuracy (R² Score): {r2_score(y_test, model.predict(X_test)):.2f}")

# A single plot showing the 'Pattern' (Study Hours vs Score)
sns.regplot(data=df, x='Study_Hours', y='Final_Score', line_kws={"color": "red"})
plt.title("Study Hours vs. Final Marks")
plt.show()

# 1. Take the inputs
h = float(input("Study Hours upto 10: "))
a = float(input("Attendance % upto 100: "))
p = float(input("Prev Score upto 100: "))
asgn = float(input("Assignments upto 10: "))

# 2. Create the input DataFrame (We call it 'user_input' here)
user_input = pd.DataFrame([[h, a, p, asgn]], columns=X.columns)

# 3. Calculate the raw math result
raw_pred = model.predict(user_input)[0]

# 4. Force the result to stay between 0 and 100
final_score = max(0, min(100, raw_pred))

print(f"\nPredicted Score: {final_score:.2f} / 100")
