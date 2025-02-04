# Deep Learning Challenge: Alphabet Soup

## Overview
The goal of this project is to build a binary classification model using deep learning techniques to predict whether applicants for Alphabet Soup funding will be successful. The model is built using TensorFlow and Keras, leveraging a dataset containing information about various organizations.

## Technologies Used
- Python
- TensorFlow/Keras
- Pandas
- Scikit-Learn
- Jupyter Notebook / Google Colab
- GitHub

## Dataset
The dataset contains metadata on over 34,000 organizations, including details such as:
- Application type
- Organization classification
- Funding use case
- Income amount
- Special considerations
- Funding amount requested
- Success status (target variable)

## Steps Completed
### 1. Data Preprocessing
- Loaded and cleaned the dataset
- Dropped unnecessary columns (EIN and NAME)
- Encoded categorical variables
- Scaled numerical data using StandardScaler
- Split data into training and testing sets

### 2. Model Development
- Created a deep learning model with TensorFlow/Keras
- Designed input, hidden, and output layers
- Compiled and trained the model
- Evaluated model accuracy and loss

### 3. Model Optimization
- Experimented with different architectures (neurons, layers, activation functions)
- Adjusted data preprocessing techniques
- Improved model performance beyond 75% accuracy

## Results
- Final model achieved a predictive accuracy of **X%**
- Key insights from training and optimization
- Possible alternative models for improvement

## Files
- `AlphabetSoupCharity.ipynb` - Initial model training and evaluation
- `AlphabetSoupCharity_Optimization.ipynb` - Optimized model
- `AlphabetSoupCharity.h5` - Saved model file
- `AlphabetSoupCharity_Optimization.h5` - Optimized model file
- `charity_data.csv` - Dataset used for training

## How to Run
1. Clone the repository:
2. Open the Jupyter Notebook or Google Colab.
3. Upload `charity_data.csv`.
4. Run the preprocessing, training, and evaluation steps.
5. Experiment with optimizations if desired.

## Future Improvements
- Implement additional feature engineering techniques
- Try alternative machine learning models (Random Forest, XGBoost, etc.)
- Further tune hyperparameters for better accuracy

## Contributors
- **Richard Encarnacion**

.

