
import os
import time
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from qiskit.circuit.library import (
    z_feature_map,
    zz_feature_map,
    pauli_feature_map
)

from qiskit_aer.primitives import SamplerV2

from qiskit_machine_learning.state_fidelities import ComputeUncompute
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.algorithms import QSVC

from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from .preprocessing import prepare_three_way_data


os.makedirs("results", exist_ok=True)


# --------------------------------
# 1. Evaluation metrics
# --------------------------------

def evaluate_model(name, y_true, predictions):

    return {
        "Model": name,
        "Accuracy": accuracy_score(y_true, predictions),
        "Precision": precision_score(
            y_true, predictions,
            average="macro",
            zero_division=0
        ),
        "Recall": recall_score(
            y_true, predictions,
            average="macro",
            zero_division=0
        ),
        "F1 Score": f1_score(
            y_true, predictions,
            average="macro",
            zero_division=0
        )
    }


# --------------------------------
# 2. Create feature maps
# --------------------------------

def create_feature_maps():

    return {
        "ZFeatureMap": z_feature_map(
            feature_dimension=4,
            reps=2
        ),

        "ZZFeatureMap": zz_feature_map(
            feature_dimension=4,
            reps=2,
            entanglement="linear"
        ),

        "PauliFeatureMap": pauli_feature_map(
            feature_dimension=4,
            reps=2,
            paulis=["Z", "ZZ", "XX"],
            entanglement="linear"
        )
    }


# --------------------------------
# 3. Save quantum circuits
# --------------------------------

def save_circuit(feature_map, name):

    circuit = feature_map.decompose(reps=2)

    filename = name.lower().replace(" ", "_")

    with open(
        f"results/{filename}_circuit.txt",
        "w",
        encoding="utf-8"
    ) as file:

        file.write(str(circuit.draw(output="text")))

    try:

        figure = circuit.draw(output="mpl", fold=-1)

        figure.savefig(
            f"results/{filename}_circuit.png",
            dpi=200,
            bbox_inches="tight"
        )

        plt.close(figure)

    except Exception as error:

        print("Circuit image could not be generated:", error)


# --------------------------------
# 4. Train quantum model
# --------------------------------

def train_quantum_model(
    feature_map, X_train, y_train, X_test
):

    sampler = SamplerV2(
        default_shots=512,
        seed=42
    )

    fidelity = ComputeUncompute(
        sampler=sampler
    )

    quantum_kernel = FidelityQuantumKernel(
        feature_map=feature_map,
        fidelity=fidelity
    )

    model = QSVC(
        quantum_kernel=quantum_kernel
    )

    start = time.time()

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    elapsed = time.time() - start

    return predictions, elapsed


# --------------------------------
# 5. Train classical SVM
# --------------------------------

def train_classical_model(
    X_train, y_train, X_test
):

    model = SVC(
        kernel="rbf",
        C=1.0
    )

    start = time.time()

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    elapsed = time.time() - start

    return predictions, elapsed


# --------------------------------
# 6. Compare models on validation data
# --------------------------------

def validate_models(
    feature_maps,
    X_train,
    y_train,
    X_val,
    y_val
):

    print("\n========== VALIDATION RESULTS ==========")

    validation_results = []

    # Test quantum feature maps
    for name, feature_map in feature_maps.items():

        print(f"\nValidating {name}...")

        predictions, elapsed = train_quantum_model(
            feature_map,
            X_train,
            y_train,
            X_val
        )

        result = evaluate_model(
            name,
            y_val,
            predictions
        )

        result["Time (seconds)"] = elapsed

        validation_results.append(result)

        print("Validation accuracy:", result["Accuracy"])
        print("Validation F1:", result["F1 Score"])

    # Test classical SVM
    print("\nValidating Classical SVM...")

    predictions, elapsed = train_classical_model(
        X_train,
        y_train,
        X_val
    )

    result = evaluate_model(
        "Classical SVM",
        y_val,
        predictions
    )

    result["Time (seconds)"] = elapsed

    validation_results.append(result)

    # Save validation results
    validation_df = pd.DataFrame(validation_results)

    validation_df.to_csv(
        "results/validation_comparison.csv",
        index=False
    )

    print("\n", validation_df.to_string(index=False))

    # Select best model using validation macro F1
    best_model = max(
        validation_results,
        key=lambda row: row["F1 Score"]
    )

    print("\nBest model based on validation F1:")
    print(best_model["Model"])

    return best_model["Model"], validation_df


# --------------------------------
# 7. Final testing
# --------------------------------

def final_testing(
    feature_maps,
    best_model_name,
    X_train,
    X_val,
    y_train,
    y_val,
    X_test,
    y_test
):

    print("\n========== FINAL TESTING ==========")

    # Combine training and validation data
    X_combined = np.concatenate(
        [X_train, X_val],
        axis=0
    )

    y_combined = np.concatenate(
        [y_train, y_val],
        axis=0
    )

    # Fit a fresh scaler on training + validation data
    scaler = MinMaxScaler()

    X_combined = scaler.fit_transform(X_combined)
    X_test = scaler.transform(X_test)

    test_results = []
    predictions_dict = {}

    # Evaluate all quantum feature maps on the test set
    for name, feature_map in feature_maps.items():

        print(f"\nTesting {name}...")

        predictions, elapsed = train_quantum_model(
            feature_map,
            X_combined,
            y_combined,
            X_test
        )

        predictions_dict[name] = predictions

        result = evaluate_model(
            name,
            y_test,
            predictions
        )

        result["Time (seconds)"] = elapsed

        test_results.append(result)

    # Evaluate classical SVM on test set
    print("\nTesting Classical SVM...")

    predictions, elapsed = train_classical_model(
        X_combined,
        y_combined,
        X_test
    )

    predictions_dict["Classical SVM"] = predictions

    result = evaluate_model(
        "Classical SVM",
        y_test,
        predictions
    )

    result["Time (seconds)"] = elapsed

    test_results.append(result)

    # Save final test comparison
    test_df = pd.DataFrame(test_results)

    test_df.to_csv(
        "results/test_comparison.csv",
        index=False
    )

    print("\n========== FINAL TEST RESULTS ==========")
    print(test_df.to_string(index=False))

    # --------------------------------
    # Confusion matrix for selected model
    # --------------------------------

    selected_predictions = predictions_dict[best_model_name]

    labels = [
        "Low support",
        "Moderate",
        "High support"
    ]

    matrix = confusion_matrix(
        y_test,
        selected_predictions,
        labels=[0, 1, 2]
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=labels
    )

    figure, ax = plt.subplots(figsize=(7, 6))

    display.plot(
        ax=ax,
        cmap="Blues",
        colorbar=False
    )

    ax.set_title(
        f"Final Test Confusion Matrix: {best_model_name}"
    )

    plt.tight_layout()

    plt.savefig(
        "results/final_confusion_matrix.png",
        dpi=200
    )

    plt.close()

    # --------------------------------
    # Test performance graph
    # --------------------------------

    metrics = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score"
    ]

    test_df.set_index("Model")[metrics].T.plot(
        kind="bar",
        figsize=(10, 6)
    )

    plt.title("Final Test Performance Comparison")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.xticks(rotation=0)
    plt.tight_layout()

    plt.savefig(
        "results/test_comparison.png",
        dpi=200
    )

    plt.close()

    # --------------------------------
    # Sample recommendations
    # --------------------------------

    recommendations = {
        0: "Provide advanced learning material and enrichment activities.",
        1: "Provide additional practice questions and learning resources.",
        2: "Provide remedial lessons and individual teacher support."
    }

    support_labels = [
        "Low",
        "Moderate",
        "High"
    ]

    print("\n========== SAMPLE PREDICTIONS ==========")
    print("Selected model:", best_model_name)

    for i, prediction in enumerate(
        selected_predictions[:5]
    ):

        print(
            f"Student {i + 1}: "
            f"{support_labels[prediction]} support required"
        )

        print(
            "Recommendation:",
            recommendations[prediction]
        )

        print()


# --------------------------------
# 8. Main function
# --------------------------------

def main():

    print("\nLoading student data...")

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test
    ) = prepare_three_way_data()

    # Fit scaler using training data only
    scaler = MinMaxScaler()

    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    print("\nTraining:", len(X_train))
    print("Validation:", len(X_val))
    print("Testing:", len(X_test))

    # Create feature maps
    feature_maps = create_feature_maps()

    # Save circuit diagrams
    for name, feature_map in feature_maps.items():

        print(f"\n{name}")
        print("Qubits:", feature_map.num_qubits)
        print(
            "Circuit depth:",
            feature_map.decompose().depth()
        )

        save_circuit(feature_map, name)

    # Select best model using validation set
    best_model_name, validation_df = validate_models(
        feature_maps,
        X_train,
        y_train,
        X_val,
        y_val
    )

    # Final evaluation on test set
    final_testing(
        feature_maps,
        best_model_name,
        X_train,
        X_val,
        y_train,
        y_val,
        X_test,
        y_test
    )

    print("\nAll experiments completed!")
    print("Check the results folder.")


if __name__ == "__main__":
    main()
