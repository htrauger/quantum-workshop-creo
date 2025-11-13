"""
Session 3: Visualization Tools for QML
Helper functions to visualize QML concepts and results
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from qiskit.visualization import plot_histogram, plot_bloch_multivector, plot_state_qsphere
from qiskit.quantum_info import Statevector
from qiskit import QuantumCircuit
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def visualize_feature_map_encoding(feature_values, feature_map_circuit):
    """
    Visualize how features are encoded in the quantum state
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Feature Map Encoding Visualization', fontsize=16)
    
    # Different feature values to test
    test_features = [
        [0.0, 0.0],
        [0.5, 0.5],
        [1.0, 0.0],
        [0.3, 0.8]
    ]
    
    for idx, features in enumerate(test_features):
        ax = axes[idx // 2, idx % 2]
        
        # Apply feature map
        qc = feature_map_circuit.bind_parameters(features)
        state = Statevector(qc)
        
        # Plot amplitude bars
        amplitudes = state.data
        positions = range(len(amplitudes))
        colors = ['red' if amp.real < 0 else 'blue' for amp in amplitudes]
        
        ax.bar(positions, np.abs(amplitudes), color=colors, alpha=0.7)
        ax.set_xlabel('Basis State')
        ax.set_ylabel('Amplitude')
        ax.set_title(f'Features: {features}')
        ax.set_xticks(positions)
        ax.set_xticklabels([f'|{bin(i)[2:].zfill(2)}⟩' for i in positions])
    
    plt.tight_layout()
    plt.savefig('feature_map_encoding.png')
    plt.show()

def plot_decision_boundary_2d(model, X, y, title="QML Decision Boundary"):
    """
    Plot 2D decision boundary for a quantum classifier
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Create mesh
    h = 0.02  # Step size in mesh
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # Predict on mesh
    mesh_data = np.c_[xx.ravel(), yy.ravel()]
    
    try:
        Z = model.predict(mesh_data)
        Z = Z.reshape(xx.shape)
        
        # Plot decision boundary
        ax.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdYlBu)
        ax.contour(xx, yy, Z, colors='black', linewidths=0.5, alpha=0.5)
    except:
        print("Could not plot decision boundary - using scatter only")
    
    # Plot data points
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, 
                        cmap=plt.cm.RdYlBu, edgecolor='black', 
                        s=100, alpha=0.8)
    
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_title(title)
    plt.colorbar(scatter, ax=ax)
    
    plt.tight_layout()
    plt.savefig('decision_boundary.png')
    plt.show()

def visualize_training_progress(loss_history, accuracy_history=None):
    """
    Visualize training progress with loss and accuracy curves
    """
    fig, axes = plt.subplots(1, 2 if accuracy_history else 1, figsize=(12, 5))
    
    if accuracy_history:
        ax1, ax2 = axes
    else:
        ax1 = axes
    
    # Plot loss
    ax1.plot(loss_history, 'b-', linewidth=2, label='Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss Over Time')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Add trend line
    z = np.polyfit(range(len(loss_history)), loss_history, 3)
    p = np.poly1d(z)
    ax1.plot(range(len(loss_history)), p(range(len(loss_history))), 
             'r--', alpha=0.5, label='Trend')
    
    # Plot accuracy if provided
    if accuracy_history:
        ax2.plot(accuracy_history, 'g-', linewidth=2, label='Training Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Training Accuracy Over Time')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        ax2.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig('training_progress.png')
    plt.show()

def plot_quantum_kernel_matrix(kernel_matrix, labels=None):
    """
    Visualize the quantum kernel matrix
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Create heatmap
    im = ax.imshow(kernel_matrix, cmap='coolwarm', aspect='auto')
    
    # Add colorbar
    plt.colorbar(im, ax=ax)
    
    # Add labels if provided
    if labels is not None:
        # Group by labels
        unique_labels = np.unique(labels)
        boundaries = []
        for label in unique_labels:
            indices = np.where(labels == label)[0]
            if len(indices) > 0:
                boundaries.append(indices[-1] + 0.5)
        
        # Draw boundaries
        for boundary in boundaries[:-1]:
            ax.axhline(boundary, color='black', linewidth=2)
            ax.axvline(boundary, color='black', linewidth=2)
    
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Sample Index')
    ax.set_title('Quantum Kernel Matrix')
    
    plt.tight_layout()
    plt.savefig('quantum_kernel_matrix.png')
    plt.show()

def compare_classical_quantum_performance(results_dict):
    """
    Create comparison plots between classical and quantum models
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Prepare data
    methods = list(results_dict.keys())
    train_scores = [results_dict[m].get('train', 0) for m in methods]
    test_scores = [results_dict[m].get('test', 0) for m in methods]
    times = [results_dict[m].get('time', 1) for m in methods]
    
    # Bar plot for accuracy
    ax1 = axes[0]
    x = np.arange(len(methods))
    width = 0.35
    bars1 = ax1.bar(x - width/2, train_scores, width, label='Train', alpha=0.8)
    bars2 = ax1.bar(x + width/2, test_scores, width, label='Test', alpha=0.8)
    
    ax1.set_xlabel('Method')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Model Performance Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods)
    ax1.legend()
    ax1.set_ylim([0, 1])
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2%}', ha='center', va='bottom')
    
    # Time comparison
    ax2 = axes[1]
    ax2.bar(methods, times, color=['blue', 'red', 'green'][:len(methods)], alpha=0.7)
    ax2.set_xlabel('Method')
    ax2.set_ylabel('Training Time (s)')
    ax2.set_title('Training Time Comparison')
    
    # Radar chart for multi-metric comparison
    ax3 = axes[2]
    categories = ['Train Acc', 'Test Acc', 'Speed']
    
    # Normalize speed (inverse of time)
    max_time = max(times) if times else 1
    speed_scores = [1 - (t/max_time) for t in times]
    
    for i, method in enumerate(methods):
        values = [train_scores[i], test_scores[i], speed_scores[i]]
        values += values[:1]  # Complete the circle
        
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]
        
        ax3.plot(angles, values, 'o-', linewidth=2, label=method)
        ax3.fill(angles, values, alpha=0.25)
    
    ax3.set_xticks(angles[:-1])
    ax3.set_xticklabels(categories)
    ax3.set_ylim([0, 1])
    ax3.set_title('Multi-metric Comparison')
    ax3.legend()
    ax3.grid(True)
    
    plt.tight_layout()
    plt.savefig('model_comparison.png')
    plt.show()

def visualize_parameter_landscape(loss_function, param_range=(-np.pi, np.pi)):
    """
    Visualize the loss landscape for 2 parameters
    """
    fig = plt.figure(figsize=(12, 5))
    
    # Create parameter grid
    theta1 = np.linspace(param_range[0], param_range[1], 50)
    theta2 = np.linspace(param_range[0], param_range[1], 50)
    Theta1, Theta2 = np.meshgrid(theta1, theta2)
    
    # Calculate loss for each point (mock function if not provided)
    if loss_function is None:
        # Mock loss landscape
        Z = np.sin(Theta1) * np.cos(Theta2) + 0.1 * Theta1**2 + 0.1 * Theta2**2
    else:
        Z = np.array([[loss_function([t1, t2]) for t2 in theta2] for t1 in theta1])
    
    # 3D surface plot
    ax1 = fig.add_subplot(121, projection='3d')
    surf = ax1.plot_surface(Theta1, Theta2, Z, cmap='viridis', alpha=0.8)
    ax1.set_xlabel('Parameter θ₁')
    ax1.set_ylabel('Parameter θ₂')
    ax1.set_zlabel('Loss')
    ax1.set_title('Loss Landscape (3D)')
    fig.colorbar(surf, ax=ax1, shrink=0.5)
    
    # 2D contour plot
    ax2 = fig.add_subplot(122)
    contour = ax2.contour(Theta1, Theta2, Z, levels=20, cmap='viridis')
    ax2.clabel(contour, inline=True, fontsize=8)
    ax2.set_xlabel('Parameter θ₁')
    ax2.set_ylabel('Parameter θ₂')
    ax2.set_title('Loss Landscape (Contour)')
    
    plt.tight_layout()
    plt.savefig('parameter_landscape.png')
    plt.show()

def plot_confusion_matrix(y_true, y_pred, class_names=None):
    """
    Create a confusion matrix visualization
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Create heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                ax=ax)
    
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title('Confusion Matrix')
    
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.show()
    
    # Print classification report
    print("\nClassification Report:")
    print("-" * 50)
    print(classification_report(y_true, y_pred, target_names=class_names))

def create_qml_architecture_diagram():
    """
    Create a diagram showing QML architecture
    """
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    
    # Define components
    components = {
        'Classical Data': (1, 4, 'lightblue'),
        'Encoding': (3, 4, 'lightgreen'),
        'Quantum Circuit': (5, 4, 'lightyellow'),
        'Measurement': (7, 4, 'lightcoral'),
        'Classical Post-processing': (9, 4, 'lightgray'),
        'Loss Function': (9, 2, 'lavender'),
        'Optimizer': (5, 2, 'lightpink'),
        'Parameter Update': (3, 2, 'lightsalmon')
    }
    
    # Draw components
    for comp, (x, y, color) in components.items():
        rect = plt.Rectangle((x-0.8, y-0.3), 1.6, 0.6, 
                            facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        ax.text(x, y, comp, ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Draw arrows
    arrows = [
        ((1.8, 4), (2.2, 4)),  # Classical -> Encoding
        ((3.8, 4), (4.2, 4)),  # Encoding -> Quantum
        ((5.8, 4), (6.2, 4)),  # Quantum -> Measurement
        ((7.8, 4), (8.2, 4)),  # Measurement -> Post-processing
        ((9, 3.7), (9, 2.3)),  # Post-processing -> Loss
        ((8.2, 2), (5.8, 2)),  # Loss -> Optimizer
        ((4.2, 2), (3.8, 2)),  # Optimizer -> Update
        ((3, 2.3), (3, 3.7)),  # Update -> Encoding (feedback)
    ]
    
    for start, end in arrows:
        ax.annotate('', xy=end, xytext=start,
                   arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    # Add labels
    ax.text(5, 5.5, 'Quantum Machine Learning Pipeline', 
            fontsize=16, fontweight='bold', ha='center')
    ax.text(5, 0.5, 'Hybrid Classical-Quantum Optimization Loop', 
            fontsize=12, ha='center', style='italic')
    
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('qml_architecture.png')
    plt.show()

if __name__ == "__main__":
    print("="*60)
    print("QML VISUALIZATION TOOLS")
    print("="*60)
    print("\nGenerating sample visualizations...")
    
    # Create sample data
    np.random.seed(42)
    X = np.random.randn(100, 2)
    y = np.random.randint(0, 2, 100)
    
    # Generate architecture diagram
    create_qml_architecture_diagram()
    print("✅ Architecture diagram created")
    
    # Sample confusion matrix
    y_true = np.random.randint(0, 2, 50)
    y_pred = np.random.randint(0, 2, 50)
    plot_confusion_matrix(y_true, y_pred, class_names=['Class 0', 'Class 1'])
    print("✅ Confusion matrix created")
    
    # Sample training progress
    loss_history = [1.0 - 0.1*i + 0.01*np.random.randn() for i in range(20)]
    accuracy_history = [0.5 + 0.02*i + 0.01*np.random.randn() for i in range(20)]
    visualize_training_progress(loss_history, accuracy_history)
    print("✅ Training progress plots created")
    
    # Sample comparison
    results = {
        'Classical SVM': {'train': 0.92, 'test': 0.88, 'time': 0.5},
        'Quantum VQC': {'train': 0.89, 'test': 0.85, 'time': 2.0},
        'Quantum Kernel': {'train': 0.91, 'test': 0.87, 'time': 1.5}
    }
    compare_classical_quantum_performance(results)
    print("✅ Comparison plots created")
    
    print("\n📊 All visualizations generated successfully!")
    print("Check the current directory for saved images.")