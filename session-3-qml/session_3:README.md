# Session 3: Quantum Machine Learning Fundamentals

## 📚 Overview
Welcome to Session 3 of the Quantum Computing Workshop! This session introduces the exciting field of Quantum Machine Learning (QML), where quantum computing meets artificial intelligence.

## 🎯 Learning Objectives
By the end of this session, you will:
- Understand the fundamentals of machine learning and how quantum computing enhances it
- Master different quantum data encoding techniques
- Implement variational quantum classifiers (VQC) and quantum kernels
- Build and train quantum neural networks
- Analyze when QML offers advantages over classical ML

## 📂 Repository Structure
```
session_3/
├── 01_setup_and_basics.py       # Setup verification and ML basics
├── 02_data_encoding.py           # Quantum data encoding techniques
├── 03_qml_algorithms.py          # Core QML algorithms (VQC, QSVC)
├── 04_hands_on_practice.py       # Interactive exercises
├── 05_visualization_tools.py     # Visualization utilities
├── 06_advanced_topics.py          # Advanced QML concepts
├── 07_utilities.py               # Helper functions
├── requirements.txt              # Dependencies
└── README.md                     # This file
```

## 🚀 Getting Started

### Prerequisites
```bash
# Install required packages
pip install -r requirements.txt
```

### Quick Setup Check
```python
python 01_setup_and_basics.py
```

## 📖 Session Flow (60 minutes)

### Part 1: Machine Learning Fundamentals (15 min)
- What is Machine Learning?
- Types of Learning (Supervised, Unsupervised, Reinforcement)
- Key ML Components
- **Script:** `01_setup_and_basics.py`

### Part 2: Introduction to QML (10 min)
- Why Quantum + ML?
- QML Categories
- Classical vs Quantum Comparison
- **Script:** `03_qml_algorithms.py` (intro section)

### Part 3: Quantum Data Encoding (10 min)
- Basis Encoding
- Amplitude Encoding
- Angle Encoding
- Advanced: IQP Encoding
- **Script:** `02_data_encoding.py`

### Part 4: QML Algorithms (10 min)
- Variational Quantum Classifier (VQC)
- Quantum Support Vector Machine (QSVM)
- Quantum Kernels
- **Script:** `03_qml_algorithms.py`

### Part 5: Hands-on Practice (10 min)
- 5 Progressive Exercises
- Build your own QML models
- **Script:** `04_hands_on_practice.py`

### Part 6: Advanced Topics & Q&A (5 min)
- Current Research Directions
- Quantum Advantage Analysis
- **Script:** `06_advanced_topics.py`

## 💻 Key Code Examples

### Example 1: Simple VQC Implementation
```python
from qiskit_machine_learning.algorithms import VQC
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes

# Create feature map and ansatz
feature_map = ZZFeatureMap(2, reps=2)
ansatz = RealAmplitudes(2, reps=2)

# Build and train VQC
vqc = VQC(feature_map=feature_map, ansatz=ansatz)
vqc.fit(X_train, y_train)
```

### Example 2: Custom Feature Encoding
```python
def angle_encoding(features):
    qc = QuantumCircuit(len(features))
    for i, feature in enumerate(features):
        qc.ry(feature * np.pi, i)
    return qc
```

### Example 3: Quantum Kernel
```python
from qiskit_machine_learning.algorithms import QSVC
from qiskit.circuit.library import PauliFeatureMap

feature_map = PauliFeatureMap(2, reps=2)
qsvc = QSVC(feature_map=feature_map)
qsvc.fit(X_train, y_train)
```

## 🏆 Exercises

### Exercise 1: Data Preparation
Normalize and prepare classical data for quantum processing.

### Exercise 2: Custom Feature Map
Design your own quantum feature encoding strategy.

### Exercise 3: Hyperparameter Tuning
Find optimal settings for QML models.

### Exercise 4: Quantum Neural Network
Build a multi-layer quantum neural network.

### Exercise 5: Real Application
Apply QML to classify real datasets.

## 📊 Visualization Tools
The session includes comprehensive visualization tools:
- Decision boundaries
- Training progress
- Quantum kernel matrices
- Performance comparisons
- Parameter landscapes

Run `05_visualization_tools.py` to generate all visualizations.

## 🔬 Advanced Topics Covered
- Barren Plateaus
- Data Re-uploading
- Quantum GANs
- Transfer Learning
- Quantum Advantage Metrics

## 📚 Additional Resources

### Papers
- [Quantum Machine Learning: What Quantum Computing Means to Data Mining](https://arxiv.org/abs/1512.02900)
- [Supervised learning with quantum-enhanced feature spaces](https://arxiv.org/abs/1804.11326)
- [The power of quantum neural networks](https://arxiv.org/abs/2011.00027)

### Online Resources
- [Qiskit Textbook - QML](https://qiskit.org/textbook/ch-machine-learning/)
- [PennyLane QML Tutorials](https://pennylane.ai/qml/)
- [IBM Quantum Network](https://quantum-computing.ibm.com/)

### Courses
- IBM Qiskit Global Summer School
- QuTech Quantum Machine Learning MOOC
- MIT xPRO Quantum Computing Fundamentals

## 🤝 Contributing
Feel free to submit issues or pull requests to improve the workshop materials.

## 📧 Contact
For questions or clarifications, please reach out during the workshop or office hours.

## ⚡ Quick Commands

```bash
# Run all scripts in sequence
python 01_setup_and_basics.py
python 02_data_encoding.py
python 03_qml_algorithms.py
python 04_hands_on_practice.py
python 05_visualization_tools.py
python 06_advanced_topics.py

# Generate all visualizations
python 05_visualization_tools.py

# Run interactive exercises only
python 04_hands_on_practice.py
```

## 🎯 Success Metrics
By completing this session, you should be able to:
- ✅ Explain the difference between classical and quantum ML
- ✅ Implement at least 3 data encoding methods
- ✅ Train a VQC on a simple dataset
- ✅ Visualize QML results
- ✅ Identify suitable use cases for QML

---
**Happy Learning! **