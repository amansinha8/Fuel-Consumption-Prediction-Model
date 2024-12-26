# Fuel-Consumption-Prediction-Model
# Fuel Efficiency and Emissions Analysis

This project involves analyzing vehicle fuel efficiency and emissions data using machine learning techniques. The dataset is cleaned, preprocessed, and used to build predictive models to estimate fuel costs, CO2 emissions, and other relevant metrics.

## Features of the Project

1. **Dataset Preprocessing**:
   - Reading data from `fuel.csv`.
   - Removing unnecessary columns to focus on relevant attributes.
   - Encoding categorical variables using `LabelEncoder`.
   - Handling missing values by dropping rows with `NaN`.

2. **Exploratory Data Analysis**:
   - Displaying dataset information (`shape`, `info`, and `describe`).
   - Checking for missing values.
   - Calculating the correlation matrix to identify highly correlated columns.

3. **Linear Regression**:
   - Building a simple linear regression model to predict `annual_fuel_cost_ft2` based on `tailpipe_co2_in_grams_mile_ft2`.
   - Building a multiple linear regression model to predict `combined_mpg_ft1` using:
     - `city_mpg_ft1`
     - `highway_mpg_ft1`
     - `unadjusted_city_mpg_ft1`
     - `unadjusted_highway_mpg_ft1`

4. **K-Nearest Neighbors (KNN) Classifier**:
   - Building a KNN classifier with `n_neighbors=3` and Euclidean distance metric to predict `annual_fuel_cost_ft2` based on `tailpipe_co2_in_grams_mile_ft2`.

5. **Support Vector Machine (SVM) Classifier**:
   - Implementing a linear SVM classifier to predict `annual_fuel_cost_ft2`.

6. **Prediction**:
   - Making predictions on new data points using trained models.
   - Example inputs are provided for various models.

7. **Evaluation**:
   - Using metrics like:
     - Mean Absolute Error (MAE)
     - R² Score for regression models.
     - Accuracy Score and Classification Report for classification models.

## Libraries Used

- `numpy`
- `pandas`
- `scikit-learn`
- `warnings`

## Code Overview

### Data Preprocessing
- Reads the dataset from `fuel.csv`.
- Drops unnecessary columns.
- Encodes categorical variables into numeric format.
- Handles missing values.

### Linear Regression
- Single-variable regression for predicting `annual_fuel_cost_ft2`.
- Multi-variable regression for predicting `combined_mpg_ft1`.

### Classification Models
- KNN classifier for `annual_fuel_cost_ft2`.
- Linear SVM classifier for `annual_fuel_cost_ft2`.

### Predictions
- Example predictions for new input data:
  - Single-variable linear regression (`tailpipe_co2_in_grams_mile_ft2`).
  - Multi-variable regression (`city_mpg_ft1`, `highway_mpg_ft1`, etc.).
  - KNN classifier.
  - SVM classifier.

## How to Run the Code

1. Ensure the dataset `fuel.csv` is available in the same directory as the code.
2. Install the required libraries using `pip install numpy pandas scikit-learn`.
3. Run the Python script to execute the models and view predictions.

## Example Output

- **Linear Regression**:
  - Coefficients, intercept, and R² score.
- **KNN Classifier**:
  - Accuracy score and predicted classes.
- **SVM Classifier**:
  - Test accuracy and predictions for new data.

## Notes

- Replace `fuel.csv` with your own dataset if needed.
- Modify hyperparameters for models like KNN (`n_neighbors`) or SVM for better performance.
- Ensure numerical data scales are compatible, especially for KNN and SVM.

## Contact

For any queries or suggestions, feel free to reach out via GitHub Issues.
