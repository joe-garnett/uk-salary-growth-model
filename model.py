import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt

smp_return = 1.08
saving_rate = 1.00
inflation = 0.025
wage_inflation = 0.059
real_wage_growth = 1.0082

# Using ONS data
x_train_ages = np.array([18, 22, 30, 40, 50, 60, 65])
base_age = min(x_train_ages)
max_age = max(x_train_ages)
y_train_salaries = np.array([24400, 27000, 35000, 40000, 38000, 30000, 25000])
y_train_adjusted_salaries = []

for age, salary in zip(x_train_ages, y_train_salaries):
    period = age - base_age
    wage_inflation_adjusted_monies = 1 * (real_wage_growth ** (age - base_age))
    invested_adjusted_monies = 1 * (smp_return ** (max_age - age))
    total_adjusted_salary = salary * wage_inflation_adjusted_monies * invested_adjusted_monies
    y_train_adjusted_salaries.append(total_adjusted_salary)



# Convert to 2D array (as learn transfomers need a number of samples, in this case 1) e.g. how many instances of 18 are there
X_train_ages = x_train_ages[:, np.newaxis]

# Create spline curve
model = make_pipeline(SplineTransformer(n_knots=4, degree=3), Ridge(alpha=1e-3))
model.fit(X_train_ages, y_train_salaries)

model_adjusted = make_pipeline(SplineTransformer(n_knots=4, degree=3), Ridge(alpha=1e-3))
model_adjusted.fit(X_train_ages, y_train_adjusted_salaries)

# Create plot for graph
x_plot = np.linspace(min(x_train_ages)-1, max(x_train_ages)+1, 100)[:, np.newaxis]
y_plot = model.predict(x_plot)
y_plot_adjusted = model_adjusted.predict(x_plot)





plt.plot(x_plot, y_plot, label="Interpolated Curve Nominal")
plt.plot(x_plot, y_plot_adjusted, label="Interpolated Curve Adjusted")
plt.scatter(x_train_ages, y_train_salaries, color='red', label="Training Points Nominal")
plt.scatter(x_train_ages, y_train_adjusted_salaries, color='black', label="Adjusted Training Points")
plt.legend()
plt.title("Median Effective Salary by Age 100% Investment rate")
plt.xlabel("Age")
plt.ylabel("Effective Salary")
plt.show()

