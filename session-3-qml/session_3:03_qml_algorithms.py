"""
Session 3: Core QML Algorithms Implementation
Variational Quantum Classifier and Quantum Kernels
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_circles
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit_machine_learning.algorithms import VQC, QSVC
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes, PauliFeatureMap
from qiskit_algorithms.optimizers import COBYLA, SPSA
from qiskit_aer import AerSimulator

def create_sample_dataset(dataset_type='moons', n_samples=100):
    """
    Create sample datasets for QML experiments
    """
    print(f"\n{'='*50}")
    print(f"Creating {dataset_type.upper()} dataset")
    print(f"{'='*50}")
    
    if dataset_type == 'moons':
        X, y = make_moons(n_samples=n_samples, noise=0.15, random_state=42)
    elif dataset_type == 'circles':
        X, y = make_circles(n_samples=n_samples, noise=0.1, factor=0.5, random_state=42)
    elif dataset_type == 'xor':
        # XOR problem
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]] * (n_samples // 4))
        y = np.array([0, 1, 1, 0] * (n_samples // 4))
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    # Normalize features to [0, 1]
    scaler = MinMaxScaler(feature_range=(0, 1))
    X = scaler.fit_transform(X)
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Dataset shape: {X.shape}")
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    print(f"Classes: {np.unique(y)}")
    
    return X_train, X_test, y_train, y_test

def build_vqc_model(feature_dimension=2, reps=2, optimizer_type='COBYLA'):
    """
    Build a Variational Quantum Classifier
    """
    print(f"\n{'='*50}")
    print("Building Variational Quantum Classifier (VQC)")
    print(f"{'='*50}")
    
    # Create feature map (encoding)
    feature_map = ZZFeatureMap(
        feature_dimension=feature_dimension,
        reps=reps,
        entanglement='linear'
    )
    
    # Create ansatz (trainable circuit)
    ansatz = RealAmplitudes(
        num_qubits=feature_dimension,
        reps=reps,
        entanglement='linear'
    )
    
    # Select optimizer
    if optimizer_type == 'COBYLA':
        optimizer = COBYLA(maxiter=100)
    elif optimizer_type == 'SPSA':
        optimizer = SPSA(maxiter=100)
    else:
        optimizer = COBYLA(maxiter=100)
    
    print(f"Feature map: {feature_map.name}")
    print(f"Feature map depth: {feature_map.depth()}")
    print(f"Ansatz: {ansatz.name}")
    print(f"Ansatz parameters: {ansatz.num_parameters}")
    print(f"Optimizer: {optimizer_type}")
    
    # Create VQC
    vqc = VQC(
        feature_map=feature_map,
        ansatz=ansatz,
        optimizer=optimizer,
        quantum_instance=AerSimulator(shots=1024)
    )
    
    return vqc, feature_map, ansatz

def train_and_evaluate_vqc(vqc, X_train, X_test, y_train, y_test):
    """
    Train and evaluate the VQC model
    """
    print(f"\n{'='*50}")
    print("Training VQC Model")
    print(f"{'='*50}")
    
    # Train the model
    print("Training in progress...")
    vqc.fit(X_train, y_train)
    print("✅ Training complete!")
    
    # Evaluate
    train_score = vqc.score(X_train, y_train)
    test_score = vqc.score(X_test, y_test)
    
    print(f"\n📊 Results:")
    print(f"Training accuracy: {train_score:.2%}")
    print(f"Testing accuracy: {test_score:.2%}")
    
    # Get predictions for analysis
    y_pred_train = vqc.predict(X_train)
    y_pred_test = vqc.predict(X_test)
    
    return train_score, test_score, y_pred_train, y_pred_test

def quantum_kernel_example(X_train, X_test, y_train, y_test):
    """
    Demonstrate Quantum Support Vector Classifier with quantum kernel
    """
    print(f"\n{'='*50}")
    print("Quantum Kernel SVM (QSVC)")
    print(f"{'='*50}")
    
    # Create quantum kernel
    feature_map = PauliFeatureMap(
        feature_dimension=2,
        reps=2,
        entanglement='full',
        paulis=['Z', 'Y', 'ZZ']
    )
    
    print(f"Quantum kernel feature map: {feature_map.name}")
    print(f"Feature map depth: {feature_map.depth()}")
    
    # Create QSVC
    qsvc = QSVC(
        feature_map=feature_map,
        quantum_instance=AerSimulator(shots=1024)
    )
    
    # Train
    print("Training QSVC...")
    qsvc.fit(X_train, y_train)
    
    # Evaluate
    train_score = qsvc.score(X_train, y_train)
    test_score = qsvc.score(X_test, y_test)
    
    print(f"\n📊 QSVC Results:")
    print(f"Training accuracy: {train_score:.2%}")
    print(f"Testing accuracy: {test_score:.2%}")
    
    return qsvc, train_score, test_score

def custom_quantum_circuit_classifier():
    """
    Build a custom quantum classifier from scratch
    """
    print(f"\n{'='*50}")
    print("Custom Quantum Classifier")
    print(f"{'='*50}")
    
    def create_custom_circuit(features, parameters):
        """
        Custom quantum circuit with specific structure
        """
        n_qubits = len(features)
        qc = QuantumCircuit(n_qubits)
        
        # Encoding layer
        for i, feature in enumerate(features):
            qc.ry(feature * np.pi, i)
        
        # Parameterized layers
        param_idx = 0
        for layer in range(2):  # 2 layers
            # Rotation layer
            for i in range(n_qubits):
                qc.ry(parameters[param_idx], i)
                param_idx += 1
                qc.rz(parameters[param_idx], i)
                param_idx += 1
            
            # Entanglement layer
            for i in range(n_qubits - 1):
                qc.cx(i, i + 1)
            if n_qubits > 2:
                qc.cx(n_qubits - 1, 0)  # Circular entanglement
        
        return qc
    
    # Example usage
    features = [0.3, 0.7]
    n_params = 2 * 2 * len(features)  # 2 rotations × 2 layers × n_qubits
    parameters = np.random.randn(n_params) * 0.1
    
    qc = create_custom_circuit(features, parameters)
    
    print(f"Custom circuit created!")
    print(f"Number of parameters: {n_params}")
    print(f"Circuit depth: {qc.depth()}")
    print(f"Number of gates: {qc.size()}")
    
    return qc

def compare_classical_vs_quantum(dataset_type='moons'):
    """
    Compare classical and quantum ML performance
    """
    print(f"\n{'='*60}")
    print("CLASSICAL vs QUANTUM COMPARISON")
    print(f"{'='*60}")
    
    # Get dataset
    X_train, X_test, y_train, y_test = create_sample_dataset(dataset_type)
    
    # Classical SVM
    from sklearn.svm import SVC
    print("\n📌 Classical SVM:")
    classical_svm = SVC(kernel='rbf')
    classical_svm.fit(X_train, y_train)
    classical_train = classical_svm.score(X_train, y_train)
    classical_test = classical_svm.score(X_test, y_test)
    print(f"Training accuracy: {classical_train:.2%}")
    print(f"Testing accuracy: {classical_test:.2%}")
    
    # Quantum VQC
    print("\n📌 Quantum VQC:")
    vqc, _, _ = build_vqc_model(feature_dimension=2, reps=1)
    quantum_train, quantum_test, _, _ = train_and_evaluate_vqc(
        vqc, X_train, X_test, y_train, y_test
    )
    
    # Results comparison
    print(f"\n{'='*60}")
    print("FINAL COMPARISON")
    print(f"{'='*60}")
    print(f"{'Method':<20} {'Train Acc':<15} {'Test Acc':<15}")
    print(f"{'-'*50}")
    print(f"{'Classical SVM':<20} {classical_train:<15.2%} {classical_test:<15.2%}")
    print(f"{'Quantum VQC':<20} {quantum_train:<15.2%} {quantum_test:<15.2%}")
    print(f"{'-'*50}")
    
    return {
        'classical': {'train': classical_train, 'test': classical_test},
        'quantum': {'train': quantum_train, 'test': quantum_test}
    }

if __name__ == "__main__":
    # Main execution
    print("="*60)
    print("QML ALGORITHMS DEMONSTRATION")
    print("="*60)
    
    # 1. Create dataset
    X_train, X_test, y_train, y_test = create_sample_dataset('moons')
    
    # 2. Build and train VQC
    vqc, feature_map, ansatz = build_vqc_model()
    train_score, test_score, _, _ = train_and_evaluate_vqc(
        vqc, X_train, X_test, y_train, y_test
    )
    
    # 3. Try quantum kernel
    qsvc, qsvc_train, qsvc_test = quantum_kernel_example(
        X_train, X_test, y_train, y_test
    )
    
    # 4. Build custom circuit
    custom_qc = custom_quantum_circuit_classifier()
    
    # 5. Compare with classical
    comparison = compare_classical_vs_quantum('moons')
    
    print("\n✅ All QML algorithms demonstrated successfully!")