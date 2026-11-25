
import qiskit
import qiskit_aer
import qiskit_machine_learning
import sklearn
import pandas
import numpy
import matplotlib
import streamlit

from src.quantum_model import main

print("======================================")
print(" QUANTUM PERSONALISED LEARNING PROJECT")
print("======================================")

print("Qiskit version:", qiskit.__version__)
print("Environment setup successful!")

print("\nStarting quantum feature map experiment...")

main()
