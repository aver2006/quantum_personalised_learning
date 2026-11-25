import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt


def load_data():

    df = pd.read_csv("dataset/student-mat.csv", sep=";")

    features = ["absences", "studytime", "G1", "G2"]
    X = df[features]

    # Target categories
    y = df["G3"].apply(
        lambda grade: 2 if grade < 10
        else 1 if grade < 15
        else 0
    )

    # Select 30 students from each category
    data = X.copy()
    data["target"] = y

    data = data.groupby(
        "target", group_keys=False
    ).sample(n=30, random_state=42)

    X = data[features].to_numpy(dtype=float)
    y = data["target"].to_numpy(dtype=int)

    return X, y


def prepare_data():

    # Original 75:25 split, retained for compatibility
    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

    scaler = MinMaxScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler


def prepare_three_way_data():

    X, y = load_data()

    # First separate 20% for final testing
    X_remaining, X_test, y_remaining, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # Divide remaining 80% into 60% training and 20% validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_remaining, y_remaining,
        test_size=0.25,
        random_state=42,
        stratify=y_remaining
    )

    print("\n========== DATA SPLIT ==========")
    print("Training records:", len(X_train))
    print("Validation records:", len(X_val))
    print("Testing records:", len(X_test))

    return X_train, X_val, X_test, y_train, y_val, y_test


if __name__ == "__main__":

    X_train, X_val, X_test, y_train, y_val, y_test = (
        prepare_three_way_data()
    )

    print("\nData preparation complete!")
    print("Features:", X_train.shape[1])
    print("Training classes:", pd.Series(y_train).value_counts().to_dict())
    print("Validation classes:", pd.Series(y_val).value_counts().to_dict())
    print("Testing classes:", pd.Series(y_test).value_counts().to_dict())
    