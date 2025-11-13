"""
Session 3: Quantum Machine Learning - Setup and Basics
Workshop: Quantum Computing with Qiskit
"""

import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram, plot_bloch_multivector
from qiskit.quantum_info import Statevector
import warnings
warnings.filterwarnings('ignore')

# Check installations
try:
    import qiskit
    from qiskit_machine_learning import datasets
    from qiskit_algorithms import optimizers
    print(f"✅ Qiskit version: {qiskit.__version__}")
    print("✅ Qiskit Machine Learning installed")
    print("✅ Ready for QML workshop!")
except ImportError as e:
    print(f"❌ Missing dependency: {e}")
    print("Please install: pip install qiskit-machine-learning qiskit-algorithms")

# Classical ML Example for comparison
def classical_ml_example():
    """Simple classical ML example to understand the basics"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    
    print("\n" + "="*50)
    print("Classical Machine Learning Example")
    print("="*50)
    
    # XOR problem - a classic non-linear problem
    X_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_train = np.array([0, 1, 1, 0])  # XOR pattern
    
    # Train classical model
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    # Predictions
    predictions = model.predict(X_train)
    accuracy = accuracy_score(y_train, predictions)
    
    print(f"Training data:\n{X_train}")
    print(f"Labels (XOR): {y_train}")
    print(f"Predictions: {predictions}")
    print(f"Accuracy: {accuracy:.2%}")
    print("\nNote: Linear classifier struggles with XOR!")
    
    return X_train, y_train

if __name__ == "__main__":
    # Run setup check
    print("="*50)
    print("QML Workshop Session 3 - Setup Check")
    print("="*50)
    
    # Run classical ML example
    X, y = classical_ml_example()