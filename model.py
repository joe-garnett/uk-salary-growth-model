import numpy as np
import streamlit as st
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt


def calc_income_tax(income):
    personal_allowance = 12_570
    basic_threshold = 50_270
    higher_threshold = 125_140
    taper_start = 100_000

    # Taper personal allowance
    if income > taper_start:
        reduction = (income - taper_start) / 2
        personal_allowance = max(0, personal_allowance - reduction)

    taxable_income = max(0, income - personal_allowance)

    tax = 0

    # Basic rate
    if taxable_income > 0:
        band = min(taxable_income, basic_threshold - personal_allowance)
        tax += band * 0.20
        taxable_income -= band

    # Higher rate
    if taxable_income > 0:
        band = min(taxable_income, higher_threshold - basic_threshold)
        tax += band * 0.40
        taxable_income -= band

    # Additional rate
    if taxable_income > 0:
        tax += taxable_income * 0.45

    return tax

def calc_ni_employee(income):
    ni = 0
    lower_threshold = 12_570
    upper_threshold = 50_270

    if income > upper_threshold:
        ni += (income - upper_threshold) * 0.02
        income = upper_threshold

    if income > lower_threshold:
        ni += (income - lower_threshold) * 0.08

    return ni

def main():
    # Economic assumptions (annual)
    investment_return_slider = st.slider("Annual Return (%)", min_value=0.0, max_value=10.0, value=0.0, step=0.5)
    price_inflation = 0.025
    investment_return = 1 + (investment_return_slider / 100) - price_inflation
    saving_return_slider = st.slider("Saving Rate (%)", min_value=0.0, max_value=100.0, value=100.0, step=1.0)
    saving_rate = saving_return_slider / 100
    wage_inflation = 0.059
    real_wage_growth = 1.0082
    income_tax = 0
    ni = 0
    apply_income_tax = st.checkbox("Apply Income Tax", False)
    apply_ni = st.checkbox("Apply Employee National Insurance", False)

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

        income_tax = calc_income_tax(nominal_salary) if apply_income_tax else 0
        ni = calc_ni_employee(nominal_salary) if apply_ni else 0

        effective_salary = (saving_rate * (nominal_salary - income_tax - ni)) * wage_growth_multiplier * investment_growth_multiplier
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

    # Find peak and lowest salary ages
    peak_i = np.argmax(salaries_adjusted_pred)
    peak_age = ages_plot[peak_i][0]
    peak_salary = salaries_adjusted_pred[peak_i]

    low_i = np.argmin(salaries_adjusted_pred)
    low_age = ages_plot[low_i][0]
    low_salary = salaries_adjusted_pred[low_i]

    # Create matplotlib figure
    fig, ax = plt.subplots()
    ax.plot(ages_plot, salaries_adjusted_pred, label="Interpolated Adjusted Salary")
    ax.scatter(peak_age, peak_salary, color='orange', label=f"Peak: £{peak_salary:,.0f} at age {peak_age:.0f}")
    ax.scatter(peak_age, low_salary, color='white', label=f"Lowest: £{low_salary:,.0f} at age {low_age:.0f}")
    ax.set_title("Median Effective Salary by Age")
    ax.set_xlabel("Age")
    ax.set_ylabel("Effective Salary * (£)")
    ax.legend()

    # Display using streamlit
    st.pyplot(fig)
    
    """
    (*) Inflation 2.5%, Wage Growth 0.82%
    """

main()
