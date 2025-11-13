"""
Session 3: Utility Functions for QML Workshop
Helper functions and utilities for the workshop exercises
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Dict, Any
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import time

def timer_decorator(func):
    """Decorator to time function execution"""
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"⏱️ {func.__name__} took {end - start:.2f} seconds")
        return result
    return wrapper

def generate_synthetic_quantum_data(n_samples: int = 100, 
                                   n_features: int = 2,
                                   quantum_noise: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic data with quantum-like properties
    """
    # Create entangled-like correlations
    X = np.random.randn(n_samples, n_features)
    
    # Add quantum-like correlations
    for i in range(1, n_features):
        X[:, i] += quantum_noise * np.sin(X[:, i-1] * np.pi)
    
    # Create non-linear separable classes
    y = np.zeros(n_samples, dtype=int)
    for i in range(n_samples):
        if np.sum(X[i]**2) % 2 < 1:
            y[i] = 1
    
    return X, y

def calculate_circuit_metrics(circuit: QuantumCircuit) -> Dict[str, Any]:
    """
    Calculate various metrics for a quantum circuit
    """
    metrics = {
        'n_qubits': circuit.num_qubits,
        'depth': circuit.depth(),
        'gate_count': circuit.size(),
        'gate_types': list(circuit.count_ops().keys()),
        'cx_count': circuit.count_ops().get('cx', 0),
        'parameter_count': circuit.num_parameters,
        'classical_bits': circuit.num_clbits
    }
    
    # Calculate two-qubit gate percentage
    two_qubit_gates = ['cx', 'cz', 'cy', 'crx', 'cry', 'crz', 'swap']
    two_qubit_count = sum(circuit.count_ops().get(gate, 0) for gate in two_qubit_gates)
    metrics['two_qubit_percentage'] = (two_qubit_count / circuit.size() * 100) if circuit.size() > 0 else 0
    
    return metrics

def print_circuit_analysis(circuit: QuantumCircuit, title: str = "Circuit Analysis"):
    """
    Print detailed analysis of a quantum circuit
    """
    print("\n" + "="*50)
    print(f"📊 {title}")
    print("="*50)
    
    metrics = calculate_circuit_metrics(circuit)
    
    for key, value in metrics.items():
        if key == 'two_qubit_percentage':
            print(f"{key.replace('_', ' ').title()}: {value:.1f}%")
        else:
            print(f"{key.replace('_', ' ').title()}: {value}")

def compare_models(models_dict: Dict[str, Any], 
                  X_test: np.ndarray, 
                  y_test: np.ndarray) -> pd.DataFrame:
    """
    Compare multiple models on the same test set
    """
    import pandas as pd
    
    results = []
    
    for name, model in models_dict.items():
        # Predict
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average='binary'
        )
        
        results.append({
            'Model': name,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1
        })
    
    return pd.DataFrame(results)

def create_animation_frames(circuit_builder, parameter_range):
    """
    Create animation frames for parameter sweep visualization
    """
    frames = []
    
    for param_value in parameter_range:
        circuit = circuit_builder(param_value)
        state = Statevector(circuit)
        frames.append(state.probabilities())
    
    return frames

def quantum_fidelity(state1: Statevector, state2: Statevector) -> float:
    """
    Calculate fidelity between two quantum states
    """
    return np.abs(state1.inner(state2)) ** 2

def save_workshop_results(results: Dict[str, Any], session: int = 3):
    """
    Save workshop results to file
    """
    import json
    from datetime import datetime
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"session_{session}_results_{timestamp}.json"
    
    # Convert numpy arrays to lists for JSON serialization
    serializable_results = {}
    for key, value in results.items():
        if isinstance(value, np.ndarray):
            serializable_results[key] = value.tolist()
        else:
            serializable_results[key] = value
    
    with open(filename, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"✅ Results saved to {filename}")

def check_quantum_advantage(quantum_score: float, 
                           classical_score: float,
                           threshold: float = 0.05) -> bool:
    """
    Check if quantum model shows advantage
    """
    advantage = quantum_score - classical_score
    
    print("\n" + "="*50)
    print("🏁 Quantum Advantage Analysis")
    print("="*50)
    print(f"Quantum Score: {quantum_score:.4f}")
    print(f"Classical Score: {classical_score:.4f}")
    print(f"Difference: {advantage:+.4f}")
    
    if advantage > threshold:
        print("✅ Quantum advantage demonstrated!")
        return True
    elif advantage > 0:
        print("📊 Quantum performs better but below threshold")
        return False
    else:
        print("❌ Classical outperforms quantum")
        return False

def generate_workshop_report(exercises_completed: List[str], 
                            scores: Dict[str, float]):
    """
    Generate a summary report for the workshop
    """
    print("\n" + "="*60)
    print("📋 WORKSHOP SUMMARY REPORT")
    print("="*60)
    
    print(f"\n✅ Exercises Completed: {len(exercises_completed)}")
    for i, exercise in enumerate(exercises_completed, 1):
        print(f"  {i}. {exercise}")
    
    if scores:
        print(f"\n📊 Performance Scores:")
        for metric, score in scores.items():
            print(f"  {metric}: {score:.2%}")
    
    # Calculate overall progress
    total_exercises = 5
    completion_rate = len(exercises_completed) / total_exercises
    
    print(f"\n🎯 Overall Progress: {completion_rate:.0%}")
    
    if completion_rate == 1.0:
        print("🎉 Congratulations! You've completed all exercises!")
    elif completion_rate >= 0.8:
        print("👏 Great work! Almost there!")
    elif completion_rate >= 0.6:
        print("💪 Good progress! Keep going!")
    else:
        print("📚 Keep practicing! You're building strong foundations!")

# Quick test utilities
def quick_qml_test():
    """
    Quick test to verify QML setup
    """
    print("Running quick QML test...")
    
    try:
        from qiskit_machine_learning.algorithms import VQC
        from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
        from sklearn.datasets import make_moons
        
        # Generate tiny dataset
        X, y = make_moons(n_samples=10, noise=0.1)
        
        # Create minimal VQC
        vqc = VQC(
            feature_map=ZZFeatureMap(2, reps=1),
            ansatz=RealAmplitudes(2, reps=1)
        )
        
        print("✅ QML setup verified!")
        return True
    except Exception as e:
        print(f"❌ QML setup issue: {e}")
        return False

if __name__ == "__main__":
    print("QML Workshop Utilities Loaded")
    print("-" * 40)
    
    # Run quick test
    if quick_qml_test():
        print("\nUtilities ready for use!")
        
        # Demo synthetic data generation
        X, y = generate_synthetic_quantum_data(50, 3)
        print(f"\nGenerated synthetic quantum data:")
        print(f"  Shape: {X.shape}")
        print(f"  Classes: {np.unique(y)}")
        
        # Demo report generation
        generate_workshop_report(
            exercises_completed=["Data Prep", "Custom Feature Map"],
            scores={"accuracy": 0.85, "f1_score": 0.83}
        )