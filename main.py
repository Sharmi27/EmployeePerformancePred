# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score


# Load the dataset
def load_data(filename):
    return pd.read_csv(filename)


# Preprocess the data
def preprocess_data(data):
    """
    Parameters:
    data (DataFrame): Input dataset.

    Returns:
    DataFrame: Preprocessed dataset.
    dict: Label encoders for categorical columns.
    StandardScaler: Scaler for numerical columns.
    """
    # Drop rows with missing values
    data = data.dropna()

    # Encode categorical variables
    label_encoders = {}
    categorical_cols = ['Education', 'City', 'Gender', 'EverBenched']
    for col in categorical_cols:
        label_encoders[col] = LabelEncoder()  # Initialize label encoder for each categorical column
        data[col] = label_encoders[col].fit_transform(data[col])  # Fit label encoder and transform the column

    # Scale numerical features
    scaler = StandardScaler()  # Initialize StandardScaler
    numerical_cols = ['JoiningYear', 'Age', 'ExperienceInCurrentDomain']
    data[numerical_cols] = scaler.fit_transform(data[numerical_cols])  # Fit scaler and transform the numerical columns
    return data, label_encoders, scaler


# Train the model
def train_model(X_train, y_train):
    """
    Train the machine learning model.

    Parameters:
    X_train (DataFrame): Features of the training data.
    y_train (Series): Target variable of the training data.

    Returns:
    XGBClassifier: Trained machine learning model.
    """
    model = XGBClassifier(n_estimators=100, random_state=42)  # Initialize XGBoost classifier
    model.fit(X_train, y_train)  # Train the model
    return model


# Evaluate the model
def evaluate_model(model, X_test, y_test):
    """
    Parameters:
    model: Trained machine learning model.
    X_test (DataFrame): Features of the test data.
    y_test (Series): Target variable of the test data.
    """
    y_pred = model.predict(X_test)  # Make predictions on the test set
    accuracy = accuracy_score(y_test, y_pred)  # Calculate accuracy
    print("Accuracy:", accuracy)  # Print accuracy


# Predict employee performance
def predict_performance(model, new_data, label_encoders, scaler):
    """
    Predict employee performance (leave or not) using the trained model.

    Parameters:
    model: Trained machine learning model.
    new_data (DataFrame): New employee data.
    label_encoders (dict): Label encoders for categorical columns.
    scaler (StandardScaler): Scaler for numerical columns.

    Returns:
    array: Predicted leave status for the new employee.
    """
    try:
        # Encode categorical variables
        for col in label_encoders:
            new_data[col] = label_encoders[col].transform(
                new_data[col])  # Transform categorical columns using label encoders

        # Scale numerical features
        new_data[new_data.columns] = scaler.transform(new_data)  # Transform numerical columns using the scaler

        prediction = model.predict(new_data)  # Make prediction for the new data
    except ValueError as e:
        # If there are previously unseen labels, return default prediction (assuming no leave)
        print(f"Warning: y contains previously unseen labels: {e}")
        unseen_col = new_data.columns[list(new_data.isna().any())]
        if len(unseen_col) > 0:
            unseen_col = unseen_col[0]
            print(f"Unseen label in column: {unseen_col}")
        else:
            print("Unseen label in unknown column")
        prediction = [0]  # Assuming no leave
    return prediction


# Main function
def main():
    # Load dataset
    filename = "employee_data.csv"
    data = load_data(filename)

    data, label_encoders, scaler = preprocess_data(data)

    # Split data into features and target variable
    X = data.drop('LeaveOrNot', axis=1)  # Features
    y = data['LeaveOrNot']  # Target variable

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # Split the data

    model = train_model(X_train, y_train)
    print("Evaluation on Test Data:")
    evaluate_model(model, X_test, y_test)

    # Predict performance for new employees
    new_employee_data = pd.DataFrame({
        'Education': ['Bachelors'],  # Education
        'JoiningYear': [2018],  # Joining Year
        'City': [2],  # City
        'Gender': [1],  # Gender
        'EverBenched': [0],  # Ever Benched
        'Age': [30],  # Age
        'ExperienceInCurrentDomain': [5]  # Experience in Current Domain
    })
    predicted_performance = predict_performance(model, new_employee_data, label_encoders, scaler)
    print("Predicted Leave Status for New Employee:", predicted_performance[0])


if __name__ == "__main__":
    main()
