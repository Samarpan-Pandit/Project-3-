# Step 1: Import required libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

#  Step 2: Load your CSV file
df = pd.read_csv("Flipkart Survey Responses Data.csv")
#  Step 3: Clean up column names (remove extra spaces)
df.columns = df.columns.str.strip()
# Step 4: Select relevant variables for causal inference
df_clean = df[['Delivery Speed Rating', 'Shopping Rating']].dropna()
# Rename columns for easier handling
df_clean.columns = ['Delivery_Speed', 'Shopping_Rating']
#  Step 5: Define treatment (X) and outcome (y)
X = df_clean[['Delivery_Speed']]   # independent variable (cause)
y = df_clean['Shopping_Rating']    # dependent variable (effect)

# 🧮 Step 6: Create and fit linear regression model
model = LinearRegression()
model.fit(X, y)
# Get the slope/coefficient (causal effect) and intercept
coef = model.coef_[0]
intercept = model.intercept_
# Predict values and compute R² score
y_pred = model.predict(X)
r2 = r2_score(y, y_pred)
# Step 7: Visualize the relationship
plt.figure(figsize=(7,5))
plt.scatter(X, y, color='blue', label='Survey Data')
plt.plot(X, y_pred, color='red', linewidth=2, label='Causal Line (Linear Fit)')
plt.xlabel('Delivery Speed Rating')
plt.ylabel('Shopping Rating')
plt.title('Causal Effect of Delivery Speed on Shopping Satisfaction')
plt.legend()
plt.grid(True)
plt.show()
# Step 8: Print interpretation
print(f"Regression Equation: Shopping_Rating = {intercept:.2f} + ({coef:.2f} × Delivery_Speed)")
print(f"R² (Model Accuracy): {r2:.3f}")
# Interpretation
if coef > 0:
    print(f"✅ Positive causal effect detected: As delivery speed increases, shopping satisfaction tends to increase by {coef:.2f} points per rating unit.")
else:
    print(f"❌ No positive causal effect: Delivery speed does not appear to increase shopping satisfaction.")