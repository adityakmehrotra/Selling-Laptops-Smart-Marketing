# Selling-Laptops-Smart-Marketing

## Project Overview

This Python project aims to predict user interest in a promotional email campaign for a laptop sale on a retail website. By analyzing historical user interaction data with similar promotions, the project utilizes a Logistic Regression model to classify users likely to be interested in the campaign. The goal is to increase campaign effectiveness by targeting the right audience and achieving high user engagement.

## Features

- **Data Integration**: Combines user demographic and behavioral data to form a comprehensive dataset.
- **Custom Preprocessing**: Implements a column transformer to handle categorical and numerical data.
- **Model Training**: Utilizes a logistic regression within a pipeline that includes preprocessing steps.
- **Predictive Analysis**: Predicts user interest based on their profiles and past interactions.

## Technology Stack

- Python 3
- Pandas for data manipulation
- GeoPandas for geographic data operations (if applicable)
- Matplotlib and Rasterio for data visualization and raster data handling
- Scikit-learn for machine learning modeling and evaluations
- SQLite for database interactions

## Getting Started

### Prerequisites

Ensure you have Python 3 installed, along with the necessary libraries:

- numpy
- pandas
- matplotlib
- geopandas
- rasterio
- scikit-learn

You can install the required libraries using the following command:

  ```bash
  pip install numpy pandas matplotlib geopandas rasterio scikit-learn
  ```

## Installation
Clone this repository to your local machine:

  ```bash
  git clone https://github.com/your-username/your-repository.git
  cd your-repository
  ```

## Usage
Run the main Python script to execute the prediction model:

  ```bash
  python main.py
  ```

This script will load the data, perform preprocessing, train the logistic regression model, and output predictions on whether users will be interested in the promotional email.

## Data Description
The dataset includes user demographics (age, badge levels), past purchase amounts, and engagement metrics (like total_visits). These features are critical for predicting user interest accurately.

## Model Details
Logistic Regression: Chosen for its efficacy in binary classification tasks.

Preprocessing: Categorical variables are one-hot encoded, while numerical variables are standardized.

Train-Test Split: The data is split into training and testing sets to ensure the model's performance is validated independently.

### Contributing
Contributions are welcome. Please fork this repository, make your changes, and submit a pull request. For major changes, please open an issue first to discuss what you would like to change.

### Last Updated: 04/12/2024
