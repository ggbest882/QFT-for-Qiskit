This programme uses libraries:

Qiskit - version 0.37.0

Numpy - version 1.26.4

Matplotlib version 3.8.3

I haven't checked on newer versions, but it should work.

The Qiskit_QFT.py program contains combined errors like depolarising noise, thermal relaxation noise, phase decay, read errors. These errors are used for basic gates, namely for two qubit operations, since QFT does not contain operations where three or more qubits are used. The parameters for the noise models are taken from an IBM quantum computer called brisbane, these are average values, pay attention to this. The programme also has an IQFT to return to the ground state and evaluate the result. The final result is written to the file where this code is located.

The Qiskit_single_noise_model_for_the_system.py program uses noise models for each system, i.e. one noise model is one system. This helps to see which errors affect the initial state the most, in order to analyse a method of minimising such errors. Errors or noise are set artificially through parameters, the values of such errors may not occur in the systems.
