
import pandas as pd


class HealthAnalyzer:
    """
    A class for analyzing health study data.
    """

    def __init__(self, data):
        """
        Initialize the analyzer with a pandas DataFrame.

        Parameters:
            data (pd.DataFrame): The health study dataset.
        """
        self.data = data

    def mean_blood_pressure(self):
        """
        Calculate the mean systolic blood pressure.

        Returns:
            float: The mean systolic blood pressure.
        """
        return self.data["systolic_bp"].mean()

    def median_blood_pressure(self):
        """
        Calculate the median systolic blood pressure.

        Returns:
            float: The median systolic blood pressure.
        """
        return self.data["systolic_bp"].median()

    def min_blood_pressure(self):
        """
        Calculate the minimum systolic blood pressure.

        Returns:
            float: The minimum systolic blood pressure.
        """
        return self.data["systolic_bp"].min()

    def max_blood_pressure(self):
        """
        Calculate the maximum systolic blood pressure.

        Returns:
            float: The maximum systolic blood pressure.
        """
        return self.data["systolic_bp"].max()
