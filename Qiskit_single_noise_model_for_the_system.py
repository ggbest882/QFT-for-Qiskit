from qiskit import QuantumCircuit, Aer, transpile, execute
from qiskit.visualization import plot_histogram
from qiskit.providers.aer.noise import NoiseModel, depolarizing_error, thermal_relaxation_error, phase_damping_error, ReadoutError
from qiskit.providers.aer.noise.errors.standard_errors import depolarizing_error
import numpy as np
import matplotlib.pyplot as plt

def initialize_state(circuit, state=5):
    binary_state = format(state, '03b')
    for i, bit in enumerate(binary_state):
        if bit == '1':
            circuit.x(i)

def qft(circuit, n):
    for j in range(n):
        circuit.h(j)
        for m in range(j+1, n):
            circuit.cp(np.pi/float(2**(m-j)), j, m)
    for qubit in range(n//2):
        circuit.swap(qubit, n-qubit-1)

def iqft(circuit, n):
    for qubit in range(n//2):
        circuit.swap(qubit, n-qubit-1)
    for j in range(n-1, -1, -1):
        circuit.h(j)
        for m in range(j-1, -1, -1):
            circuit.cp(-np.pi/float(2**(j-m)), m, j)

def create_thermal_relaxation_error(t1, t2, gate_time, num_qubits):
    thermal_error = depolarizing_error(gate_time / t1, 1)
    return thermal_error
    
n=3
# Параметры шума
t1 = 75
t2 = 37.5
gate_time = 0.5
depolarizing_error_rate = 0.2
phase_damping_rate = 0.2
measurement_error_probability = 0.2

def create_noise_model(noise_type):
    noise_model = NoiseModel()
    if noise_type == "depolarizing":
        dep_error_1q = depolarizing_error(depolarizing_error_rate, 1)
        dep_error_2q = depolarizing_error(depolarizing_error_rate, 2)
        gates_1q = ['u1', 'u2', 'u3', 'rz', 'sx', 'h']
        gates_2q = ['cp', 'swap']
        for gate in gates_1q:
            for qubit in range(n):
                noise_model.add_quantum_error(dep_error_1q, gate, [qubit])
        for gate in gates_2q:
            for qubit in range(n):
                for qubit2 in range(qubit+1, n):
                    noise_model.add_quantum_error(dep_error_2q, gate, [qubit, qubit2])
    elif noise_type == "thermal":
        thermal_error_1q = create_thermal_relaxation_error(t1, t2, gate_time, 1)
        gates_1q = ['u1', 'u2', 'u3', 'rz', 'sx', 'h']
        for gate in gates_1q:
            for qubit in range(n):
                noise_model.add_quantum_error(thermal_error_1q, gate, [qubit])
    elif noise_type == "phase damping":
        phase_error_1q = phase_damping_error(phase_damping_rate)
        gates_1q = ['u1', 'u2', 'u3', 'rz', 'sx', 'h']
        for gate in gates_1q:
            for qubit in range(n):
                noise_model.add_quantum_error(phase_error_1q, gate, [qubit])
    elif noise_type == "readout":
        readout_error = ReadoutError([[1 - measurement_error_probability, measurement_error_probability], 
                                      [measurement_error_probability, 1 - measurement_error_probability]])
        for qubit in range(n):
            noise_model.add_readout_error(readout_error, [qubit])
    return noise_model


noise_types = ["depolarizing", "thermal", "phase damping", "readout"]
backend = Aer.get_backend('qasm_simulator')

for noise_type in noise_types:
    # Создание новой схемы для каждого эксперимента
    qc = QuantumCircuit(n)
    initialize_state(qc, 5)
    iqft(qc, n)
    qft(qc, n)
    qc.measure_all()

    noise_model = create_noise_model(noise_type)
    transpiled_qc = transpile(qc, backend, optimization_level=0)
    #transpiled_qc.draw(output='mpl')
    job = execute(transpiled_qc, backend, noise_model=noise_model, shots=4000)
    result = job.result()
    counts = result.get_counts(transpiled_qc)

    full_counts_with_noise = {format(i, '03b'): 0 for i in range(2**n)}
    full_counts_with_noise.update(counts)

    # Вывод гистограммы
    plot_histogram(full_counts_with_noise, title=f'A histogram with {noise_type} noise', bar_labels=True)
    plt.show()