from qiskit import QuantumCircuit, Aer, transpile, execute
from qiskit.visualization import plot_histogram
from qiskit.providers.aer.noise import NoiseModel, depolarizing_error, thermal_relaxation_error, phase_damping_error, ReadoutError
import numpy as np
import matplotlib.pyplot as plt

def initialize_state(circuit, state=5):
    binary_state = format(state, '05b')
    for i, bit in enumerate(reversed(binary_state)):
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

n = 5 # Qubit count for our system

# Parameters for the noise model:
t1 = 218.04
t2 = 127.4
gate_time = 0.15
depolarizing_error_rate = 0.07888
phase_damping_rate = 0.01
measurement_error_probability = 0.02003

def create_noise_model():
    noise_model = NoiseModel()

    # Noise type
    dep_error_1q = depolarizing_error(depolarizing_error_rate, 1)
    thermal_error_1q = thermal_relaxation_error(t1, t2, gate_time)
    phase_error_1q = phase_damping_error(phase_damping_rate)
    readout_error = ReadoutError([[1 - measurement_error_probability, measurement_error_probability], 
                                  [measurement_error_probability, 1 - measurement_error_probability]])

    # Noise list
    gates_1q = ['u1', 'u2', 'u3', 'rz', 'sx', 'h']

    # Plus noise together
    for qubit in range(n):
        for gate in gates_1q:
            # Combinate all noise in one
            combined_error = dep_error_1q.compose(thermal_error_1q).compose(phase_error_1q)
            noise_model.add_quantum_error(combined_error, gate, [qubit])
    
    # Plus noise readouts
    for qubit in range(n):
        noise_model.add_readout_error(readout_error, [qubit])

    return noise_model

backend = Aer.get_backend('qasm_simulator')
qc = QuantumCircuit(n)
initialize_state(qc, 5)
qft(qc, n)
iqft(qc, n)
qc.measure_all()

noise_model = create_noise_model()
transpiled_qc = transpile(qc, backend, optimization_level=0)
job = execute(transpiled_qc, backend, noise_model=noise_model, shots=2048) # parametr SHOTS you can change for testing on your opinion, 
result = job.result()
counts = result.get_counts(transpiled_qc)

full_counts_with_noise = {format(i, '05b'): 0 for i in range(2**n)}
full_counts_with_noise.update(counts)
plot_histogram(full_counts_with_noise, title='A histogram with combined noise', bar_labels=True)
plt.show()
