"""
Session 3: Quantum Data Encoding Techniques
Demonstrates different ways to encode classical data into quantum states
"""

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram, circuit_drawer
from qiskit.quantum_info import Statevector
import matplotlib.pyplot as plt

def basis_encoding(data):
    """
    Basis Encoding: Binary representation
    Each bit becomes a qubit state |0⟩ or |1⟩
    """
    print("\n" + "="*50)
    print("BASIS ENCODING")
    print("="*50)
    
    # Convert integer to binary
    binary = format(data, '03b')  # 3-bit representation
    n_qubits = len(binary)
    
    print(f"Encoding integer: {data}")
    print(f"Binary representation: {binary}")
    
    # Create circuit
    qc = QuantumCircuit(n_qubits, n_qubits)
    
    # Apply X gates where bit is 1
    for i, bit in enumerate(binary):
        if bit == '1':
            qc.x(i)
    
    qc.measure_all()
    
    print(f"Circuit depth: {qc.depth()}")
    print(f"Number of qubits needed: {n_qubits}")
    
    return qc

def amplitude_encoding(data_vector):
    """
    Amplitude Encoding: Encode data as amplitudes
    N data points → log2(N) qubits
    """
    print("\n" + "="*50)
    print("AMPLITUDE ENCODING")
    print("="*50)
    
    # Normalize the data vector
    norm = np.linalg.norm(data_vector)
    normalized_data = data_vector / norm if norm != 0 else data_vector
    
    print(f"Original data: {data_vector}")
    print(f"Normalized data: {normalized_data}")
    
    # Calculate required qubits
    n_qubits = int(np.ceil(np.log2(len(data_vector))))
    
    # Pad with zeros if necessary
    padded_length = 2**n_qubits
    if len(normalized_data) < padded_length:
        normalized_data = np.pad(normalized_data, (0, padded_length - len(normalized_data)))
    
    # Create circuit
    qc = QuantumCircuit(n_qubits)
    qc.initialize(normalized_data, range(n_qubits))
    
    print(f"Data points: {len(data_vector)}")
    print(f"Qubits needed: {n_qubits}")
    print(f"Compression ratio: {len(data_vector)/n_qubits:.2f}x")
    
    return qc, normalized_data

def angle_encoding(features, encoding_type='ry'):
    """
    Angle Encoding: Encode features as rotation angles
    Each feature becomes a rotation angle
    """
    print("\n" + "="*50)
    print("ANGLE ENCODING")
    print("="*50)
    
    n_qubits = len(features)
    qc = QuantumCircuit(n_qubits)
    
    print(f"Features to encode: {features}")
    print(f"Encoding type: {encoding_type.upper()} rotations")
    
    for i, feature in enumerate(features):
        angle = feature * np.pi  # Scale to [0, π]
        if encoding_type == 'ry':
            qc.ry(angle, i)
        elif encoding_type == 'rz':
            qc.rz(angle, i)
        elif encoding_type == 'rx':
            qc.rx(angle, i)
    
    print(f"Number of qubits: {n_qubits}")
    print(f"Circuit depth: {qc.depth()}")
    
    return qc

def iqp_encoding(features):
    """
    IQP (Instantaneous Quantum Polynomial) Encoding
    Creates entanglement between features
    """
    print("\n" + "="*50)
    print("IQP ENCODING (Advanced)")
    print("="*50)
    
    n_qubits = len(features)
    qc = QuantumCircuit(n_qubits)
    
    # First layer: Hadamard gates
    for i in range(n_qubits):
        qc.h(i)
    
    # Encode features as phases
    for i, feature in enumerate(features):
        qc.p(feature * np.pi, i)
    
    # Entangling layer
    for i in range(n_qubits - 1):
        for j in range(i + 1, n_qubits):
            qc.cp(features[i] * features[j] * np.pi, i, j)
    
    print(f"Features: {features}")
    print(f"Creates feature interactions: x_i * x_j")
    print(f"Circuit depth: {qc.depth()}")
    
    return qc

def visualize_encodings():
    """
    Visualize and compare different encoding methods
    """
    print("\n" + "="*60)
    print("COMPARING ENCODING METHODS")
    print("="*60)
    
    # Sample data
    integer_data = 5
    vector_data = np.array([0.5, 0.3, 0.8, 0.2])
    features = [0.3, 0.7]
    
    # Create all encodings
    basis_qc = basis_encoding(integer_data)
    amp_qc, _ = amplitude_encoding(vector_data)
    angle_qc = angle_encoding(features)
    iqp_qc = iqp_encoding(features)
    
    # Comparison table
    print("\n" + "-"*60)
    print("ENCODING METHOD COMPARISON")
    print("-"*60)
    print(f"{'Method':<20} {'Qubits':<10} {'Depth':<10} {'Best For':<30}")
    print("-"*60)
    print(f"{'Basis':<20} {'N bits':<10} {'1':<10} {'Integer/Binary data':<30}")
    print(f"{'Amplitude':<20} {'log₂(N)':<10} {'O(2^n)':<10} {'Dense vectors':<30}")
    print(f"{'Angle':<20} {'N':<10} {'1':<10} {'Continuous features':<30}")
    print(f"{'IQP':<20} {'N':<10} {'2-3':<10} {'Feature interactions':<30}")
    print("-"*60)
    
    return basis_qc, amp_qc, angle_qc, iqp_qc

def interactive_encoding_demo():
    """
    Interactive demo where users can input their own data
    """
    print("\n" + "="*60)
    print("INTERACTIVE ENCODING DEMO")
    print("="*60)
    print("Try encoding your own data!")
    
    # Example with user-like input
    user_features = [0.25, 0.75, 0.5]
    print(f"\nEncoding features: {user_features}")
    
    # Create circuit with angle encoding
    qc = QuantumCircuit(len(user_features))
    
    for i, feat in enumerate(user_features):
        qc.ry(feat * np.pi, i)
    
    # Add entanglement
    for i in range(len(user_features) - 1):
        qc.cx(i, i + 1)
    
    # Get the statevector
    backend = AerSimulator(method='statevector')
    statevector = Statevector(qc)
    
    print("\nResulting quantum state amplitudes:")
    for i, amp in enumerate(statevector.data[:8]):  # Show first 8 amplitudes
        print(f"|{format(i, '03b')}⟩: {amp:.3f}")
    
    return qc

if __name__ == "__main__":
    # Run all encoding demonstrations
    circuits = visualize_encodings()
    
    # Run interactive demo
    interactive_qc = interactive_encoding_demo()
    
    print("\n✅ All encoding methods demonstrated!")
    print("📝 Note: Check the repository for visualization notebooks")