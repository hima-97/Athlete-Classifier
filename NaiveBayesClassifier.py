import sys
import math
import pandas as pd

# Column names for the dataset
column_names = ["age", "gender", "height_cm", "weight_kg", "body_fat_%", "diastolic", "systolic", "grip_force", 
                "sit_and_bend_forward_cm", "sit_up_count", "broad_jump_cm", "label"]

# Selected features for the classifier
features = ["sit_and_bend_forward_cm", "sit_up_count"]

def load_data(file_path, has_labels=True):
    """
    Load data from a CSV file and assign column names.
    Args:
    - file_path: path to the CSV file.
    - has_labels: boolean indicating if the file has label column.
    Returns:
    - data: pandas DataFrame with the loaded data.
    """
    try:
        data = pd.read_csv(file_path, header=None)
        data.columns = column_names[:len(data.columns)] if has_labels else column_names[:-1]
        return data
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

def preprocess_data(data):
    """
    Preprocess the data by segregating it based on labels and gender.
    Args:
    - data: pandas DataFrame with the loaded data.
    Returns:
    - Dictionary with segregated data based on gender and label.
    """
    return {
        'm_model_0': data[(data["label"] == 0) & (data["gender"] == 'M')],
        'm_model_1': data[(data["label"] == 1) & (data["gender"] == 'M')],
        'f_model_0': data[(data["label"] == 0) & (data["gender"] == 'F')],
        'f_model_1': data[(data["label"] == 1) & (data["gender"] == 'F')]
    }

def train_model(segregated_data):
    """
    Train the model by calculating the mean and variance for each feature.
    Args:
    - segregated_data: dictionary with segregated data.
    Returns:
    - models: dictionary with mean and variance for each feature in each class.
    """
    models = {}
    for key, df in segregated_data.items():
        models[key] = {
            'mean': df[features].mean(),
            'variance': df[features].var()
        }
    return models

def evaluate_model(test_data, models):
    """
    Evaluate the model on the test data.
    Args:
    - test_data: pandas DataFrame with the test data.
    - models: dictionary with trained models.
    Returns:
    - test_data: pandas DataFrame with the predicted labels added.
    """
    test_features = test_data[features].copy()

    # Calculate the log-probability scores for each feature and each class
    for feature in features:
        for model_key, model_stats in models.items():
            mean_val = model_stats['mean'][feature]
            var_val = model_stats['variance'][feature]
            score_column = f"{feature}_{model_key}"
            test_features[score_column] = test_features[feature].apply(
                lambda x: -math.log(var_val) - ((x - mean_val) ** 2) / (2 * var_val))

    # Sum the log-probability scores for each class
    for gender in ['m', 'f']:
        for label in [0, 1]:
            score_cols = [f"{feature}_{gender}_model_{label}" for feature in features]
            test_features[f"score_{gender}{label}"] = test_features[score_cols].sum(axis=1)

    # Determine the predicted class based on the highest score
    test_data["m_predicted"] = (test_features["score_m1"] > test_features["score_m0"]).astype(int)
    test_data["f_predicted"] = (test_features["score_f1"] > test_features["score_f0"]).astype(int)
    test_data["predicted"] = test_data.apply(
        lambda row: row["m_predicted"] if row["gender"] == 'M' else row["f_predicted"], axis=1)

    return test_data

def print_predictions(test_data):
    """
    Print the predicted labels.
    Args:
    - test_data: pandas DataFrame with the predicted labels.
    """
    for predicted in test_data["predicted"].to_list():
        print(predicted)

def compute_accuracy(test_data):
    """
    Compute and print the accuracy of the predictions.
    Args:
    - test_data: pandas DataFrame with the predicted and actual labels.
    """
    if 'label' in test_data.columns:
        test_data["accuracy"] = (test_data["label"] == test_data["predicted"]).astype(int)
        accuracy = test_data["accuracy"].mean() * 100
        print(f"Accuracy: {accuracy:.2f}%")

def main():
    if len(sys.argv) != 3:
        print("Usage: python NaiveBayesClassifier.py <training_file> <testing_file>")
        sys.exit(1)

    train_file_path = sys.argv[1]
    test_file_path = sys.argv[2]

    # Load and preprocess the training data
    train_data = load_data(train_file_path)
    segregated_data = preprocess_data(train_data)
    models = train_model(segregated_data)

    # Load and evaluate the test data
    test_data = load_data(test_file_path, has_labels=False)
    evaluated_test_data = evaluate_model(test_data, models)
    
    # Print predictions and compute accuracy
    print_predictions(evaluated_test_data)
    compute_accuracy(evaluated_test_data)

if __name__ == "__main__":
    main()
