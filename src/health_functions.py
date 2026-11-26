
import numpy as np
import scipy.stats as stats


def calculate_disease_prevalence(data):
    """
    Calculate the prevalence of disease in the dataset.

    Parameters:
        data (pandas.DataFrame): The health study dataset.

    Returns:
        float: The proportion of individuals with disease.
    """
    total_people = len(data)
    disease_cases = data["disease"].sum()

    prevalence = disease_cases / total_people

    return prevalence


def calculate_confidence_interval(data, confidence=0.95):
    """
    Calculate the confidence interval for mean systolic blood pressure.

    Parameters:
        data (pandas.DataFrame): The health study dataset.
        confidence (float): The confidence level.

    Returns:
        tuple: The lower and upper confidence interval bounds.
    """
    blood_pressure = data["systolic_bp"]

    mean_bp = blood_pressure.mean()
    std_bp = blood_pressure.std()
    n = len(blood_pressure)

    standard_error = std_bp / np.sqrt(n)

    t_critical = stats.t.ppf(
        (1 + confidence) / 2,
        df=n - 1
    )

    margin_of_error = t_critical * standard_error

    lower_bound = mean_bp - margin_of_error
    upper_bound = mean_bp + margin_of_error

    return lower_bound, upper_bound


def calculate_linear_regression(data):
    """
    Calculate linear regression between age and systolic blood pressure.

    Parameters:
        data (pandas.DataFrame): The health study dataset.

    Returns:
        tuple: The slope and intercept of the regression line.
    """
    x = data["age"].to_numpy()
    y = data["systolic_bp"].to_numpy()

    slope, intercept = np.polyfit(x, y, 1)

    return slope, intercept


def calculate_r_squared(data, slope, intercept):
    """
    Calculate the R-squared value for the linear regression model.

    Parameters:
        data (pandas.DataFrame): The health study dataset.
        slope (float): The regression slope.
        intercept (float): The regression intercept.

    Returns:
        float: The coefficient of determination (R-squared).
    """
    x = data["age"].to_numpy()
    y = data["systolic_bp"].to_numpy()

    predicted_y = intercept + slope * x

    ss_total = np.sum((y - np.mean(y)) ** 2)
    ss_residual = np.sum((y - predicted_y) ** 2)

    r_squared = 1 - (ss_residual / ss_total)

    return r_squared