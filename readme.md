## FFNN to solve the Schrodinger Equation 
Uses Fourier transform and a FFNN to approximate the ground state solution to the Schrodinger's equation for a 1d generic potential V(x). The model was built using Pytorch and can be trained with randomly generated potentials.

### Generating the potentials
The potentials to train and test the network can be generated using ```create_dataset.py```. The number of batches of functions generated and the size of the batches is decided in ```network/config.py```. 
The following example generates a testing set in ```network/data/testing.pt``` using the gpu, overwriting the currently existing one.
```bash
python create_dataset.py -T "network/data/testing.pt" -go
```

### Training and testing the network
The network can be generated, trained and tested using ```use_network.py```. The number of parameters and layers can be modified in the file ```network/config.py```. An example already trained network is already provided in ```network/data/checkpoint.pth```. The following example tests this network with the testing set ```network/data/testing.pt``` and plots 5 graphs showing how the network performed with 5 of the potentials in the set.
```bash
python use_network.py -m "network/data/checkpoint.pth" -T "network/data/testing.pt" -p 5
```
