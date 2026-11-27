1) Overview

This project explores Quantum Machine Learning (QML) for classifying students into academic performance-based support categories.

It compares three quantum kernel-based Support Vector Classifiers (QSVCs), using different quantum feature maps, against a classical Support Vector Machine (SVM).

The objective is to investigate whether quantum feature representations offer useful classification performance on a small student performance dataset.


2) Objectives

1. Explore quantum feature maps for machine learning.
2. Implement quantum kernel-based classification using Qiskit.
3. Compare quantum models with a classical SVM baseline.
4. Evaluate model performance using validation and independent test sets.


3) Dataset

The project uses the [UCI Student Performance Dataset](https://archive.ics.uci.edu/dataset/320/student+performance), specifically the Mathematics (`student-mat.csv`) dataset.

The original dataset contains 395 student records and 33 attributes.

Four input features were selected:

| Feature | Description |
|---|---|
| `absences` | Number of school absences |
| `studytime` | Weekly study time category |
| `G1` | First-period grade |
| `G2` | Second-period grade |

The final grade (`G3`) is used only to construct the target labels. It is not provided as a model input.

3.1) Target Classes

| Class | Support Category | Condition |
|---|---|---|
| 0 | Low support | G3 ≥ 15 |
| 1 | Moderate support | 10 ≤ G3 < 15 |
| 2 | High support | G3 < 10 |

These categories are proxies derived from final academic performance, not directly observed student support requirements.

A balanced subset of 90 records was created, with 30 records per class. This smaller dataset was used to keep quantum kernel simulation computationally manageable.


4) Methodology

1. Load and preprocess the dataset.
2. Select four input features and construct target labels from G3.
3. Create a balanced subset of 90 records.
4. Perform a stratified 60:20:20 train-validation-test split:
   - Training: 54 records
   - Validation: 18 records
   - Testing: 18 records
5. Fit preprocessing transformations using training data.
6. Train and compare candidate models using validation macro F1-score.
7. Select the best model based on validation performance.
8. Evaluate all candidate models on the held-out test set.

For final testing, the candidate models are retrained using the combined training and validation sets. The scaler is refitted on this combined dataset before transforming the training and test data.


5) Quantum Machine Learning Approach

A quantum feature map transforms classical input data into quantum states.

For an input vector $x$, the feature map prepares a quantum state $|\phi(x)\rangle$. The quantum kernel measures the squared overlap between two such states:

$$
K(x,x') = |\langle\phi(x)|\phi(x')\rangle|^2
$$

These kernel values are used by a Support Vector Classifier to perform classification.

5.1) Quantum Circuit Configuration

1. Number of qubits: 4 (one per selected feature)
2. Repetitions: 2
3. Simulator: Qiskit Aer
4. Shots: 512
5. Random seed: 42

Three feature maps were evaluated:

| Feature Map | Configuration |
|---|---|
| ZFeatureMap | 4 features, 2 repetitions |
| ZZFeatureMap | 4 features, 2 repetitions, linear entanglement |
| PauliFeatureMap | Pauli operators: Z, ZZ, XX; 2 repetitions |

The quantum kernel was implemented using `FidelityQuantumKernel` and `ComputeUncompute`.

The quantum circuits were executed using a classical simulator, not physical quantum hardware.


6) Classical Baseline

A classical Support Vector Machine with an RBF kernel was used as the baseline.

$$
K(x,x') = e^{-\gamma\|x-x'\|^2}
$$

This provides a comparison between quantum fidelity-based kernels and a conventional nonlinear kernel.


7) Results

7.1) Validation and Test Performance

Macro F1-score was used as the primary model-selection metric.

| Model | Validation Macro F1 | Test Macro F1 |
|---|---:|---:|
| ZFeatureMap QSVC | 0.8322 | 0.8857 |
| ZZFeatureMap QSVC | 0.7304 | 0.6084 |
| PauliFeatureMap QSVC | 0.5519 | 0.6256 |
| Classical RBF-SVM | **0.8889** | 0.8322 |

The Classical RBF-SVM was selected based on its highest validation macro F1-score.

7.2) Final Confusion Matrix

![Final Confusion Matrix](results/final_confusion_matrix.png)

Additional performance comparisons, confusion matrices and circuit visualisations are available in the `results/` directory.


8) Findings

The classical RBF-SVM achieved the highest validation macro F1-score (0.8889), while the ZFeatureMap QSVC achieved the highest test macro F1-score (0.8857).

Model selection was based on validation performance rather than test performance. The results show that the classical baseline performed best during model selection on this dataset. They do not establish quantum advantage.

Interestingly, the ZFeatureMap QSVC performed better on the test set than the two feature maps incorporating entangling operations (ZZ and Pauli). This suggests that increased circuit complexity or entanglement does not automatically improve classification performance, particularly with a small dataset and limited training samples.


9) Limitations

1. The experiment uses only 90 records from the original dataset.
2. The test set contains 18 records, making performance estimates sensitive to individual predictions. One test prediction changes accuracy by approximately 5.56 percentage points.
3. Only one train-validation-test split has been evaluated so far. Repeated splits are needed to assess result stability.
4. G1 and G2 are strongly related to G3, which may make the classification task easier. An ablation study excluding these features would help assess their influence.
5. The target categories are derived from final grades and do not represent independently measured student support needs.
6. Quantum models were evaluated using classical simulation rather than actual quantum hardware.
7. No quantum advantage is claimed.


10) Technologies Used

1. Python 3.13.3
2. Qiskit 2.6.2
3. Qiskit Aer
4. Qiskit Machine Learning
5. Scikit-learn
6. Pandas
7. NumPy
8. Matplotlib
9. Streamlit


11) Reproducibility

The experiments use random seed `42` for reproducibility.

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the experiment:

```bash
python main.py
```

Launch the Streamlit application:

```bash
streamlit run app.py
```


12) Future Work

1. Evaluate model stability across multiple random splits and report mean and standard deviation.
2. Conduct feature ablation experiments, particularly excluding G1 and G2.
3. Explore larger datasets and additional feature maps.
4. Investigate execution on actual quantum hardware.
5. Improve the interactive Streamlit application.


13) References

13.1) Dataset

1. Cortez, P., & Silva, A. (2008). Using data mining to predict secondary school student performance. In *Proceedings of the 5th Future Business Technology Conference (FUBUTEC 2008)*, pp. 5–12. Porto, Portugal: EUROSIS.

2. Cortez, P. (2008). *Student Performance* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5TG7T

13.2) Quantum Machine Learning

3. Havlíček, V., Córcoles, A. D., Temme, K., Harrow, A. W., Kandala, A., Chow, J. M., & Gambetta, J. M. (2019). Supervised learning with quantum-enhanced feature spaces. *Nature, 567*, 209–212. https://doi.org/10.1038/s41586-019-0980-2

4. Schuld, M., & Killoran, N. (2019). Quantum machine learning in feature Hilbert spaces. *Physical Review Letters, 122*(4), 040504. https://doi.org/10.1103/PhysRevLett.122.040504

13.3) Framework and Implementation

5. Qiskit Machine Learning Documentation. *Quantum Kernel Machine Learning*. https://qiskit-community.github.io/qiskit-machine-learning/tutorials/03_quantum_kernel.html

6. Qiskit Machine Learning Documentation. *FidelityQuantumKernel API Reference*. https://qiskit-community.github.io/qiskit-machine-learning/stubs/qiskit_machine_learning.kernels.FidelityQuantumKernel.html

7. Scikit-learn Documentation. *Support Vector Classification (SVC)*. https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
