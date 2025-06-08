import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt

# Economic assumptions (annual)
investment_return = 1.08 
saving_rate = 1.00 
price_inflation = 0.025
wage_inflation = 0.059
real_wage_growth = 1.0082

# ONS salary data
ages = np.array([18, 22, 30, 40, 50, 60, 65])
salaries_nominal = np.array([24400, 27000, 35000, 40000, 38000, 30000, 25000])

# Setup
min_age = min(ages)
max_age = max(ages)
salaries_adjusted = []

# Calc wage inflation and investment adjusted salary
for age, nominal_salary in zip(ages, salaries_nominal):
    years_since_base = age - min_age
    years_until_retirement = max_age - age

    wage_growth_multiplier = real_wage_growth ** years_since_base
    investment_growth_multiplier = investment_return ** years_until_retirement

    effective_salary = (saving_rate * nominal_salary) * wage_growth_multiplier * investment_growth_multiplier
    salaries_adjusted.append(effective_salary)

# Convert to 2D arrays for sklearn transformers
ages_2d = ages[:, np.newaxis]

# Create spline models for nominal and adjusted salaries
model_nominal = make_pipeline(SplineTransformer(n_knots=4, degree=3), Ridge(alpha=1e-3))
model_nominal.fit(ages_2d, salaries_nominal)

model_adjusted = make_pipeline(SplineTransformer(n_knots=4, degree=3), Ridge(alpha=1e-3))
model_adjusted.fit(ages_2d, salaries_adjusted)

# Plot range
ages_plot = np.linspace(min_age - 1, max_age + 1, 100)[:, np.newaxis]
salaries_nominal_pred = model_nominal.predict(ages_plot)
salaries_adjusted_pred = model_adjusted.predict(ages_plot)

# Plot
plt.plot(ages_plot, salaries_nominal_pred, label="Interpolated Nominal Salary")
plt.plot(ages_plot, salaries_adjusted_pred, label="Interpolated Adjusted Salary")
plt.scatter(ages, salaries_nominal, color='red', label="Nominal Salary Data")
plt.scatter(ages, salaries_adjusted, color='black', label="Adjusted Salary Data")
plt.legend()
plt.title("Median Effective Salary by Age (100% Investment Rate)")
plt.xlabel("Age")
plt.ylabel("Effective Salary (£)")
plt.show()
