import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression

# Download and prepare the data
data_root = "https://github.com/ageron/data/raw/main/"
life_sat = pd.read_csv(data_root+'lifesat/lifesat.csv')
life_sat.columns = life_sat.columns.str.replace(' ,', '')
X = life_sat[["GDP per capita (USD)"]].values
Y = life_sat[["Life satisfaction"]].values

# Create and fit a linear regression model
model = LinearRegression()
model.fit(X, Y)

# Plot the scatter plot with the line of best fit
plt.scatter(X, Y, s=10)
plt.plot(X, model.predict(X), color='r')
plt.xlabel("GDP per capita (USD)")
plt.ylabel("Life satisfaction")
plt.grid(True)
plt.show()
