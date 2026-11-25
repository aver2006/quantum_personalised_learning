
from qiskit.circuit.library import zz_feature_map
from qiskit_aer.primitives import SamplerV2

from qiskit_machine_learning.state_fidelities import ComputeUncompute
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.algorithms import QSVC

from sklearn.svm import SVC

from preprocessing import prepare_data


def train_models():
    X_train, X_test, y_train, y_test, scaler = prepare_data()

    feature_map = zz_feature_map(
        feature_dimension=4,
        reps=2,
        entanglement="linear"
    )

    sampler = SamplerV2(default_shots=512, seed=42)
    fidelity = ComputeUncompute(sampler=sampler)

    quantum_kernel = FidelityQuantumKernel(
        feature_map=feature_map,
        fidelity=fidelity
    )

    print("Training Quantum SVC...")
    quantum_model = QSVC(quantum_kernel=quantum_kernel)
    quantum_model.fit(X_train, y_train)

    print("Training Classical SVM...")
    classical_model = SVC(kernel="rbf", C=1.0)
    classical_model.fit(X_train, y_train)

    print("Both models are ready!")

    return quantum_model, classical_model, scaler
