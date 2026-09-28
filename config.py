### Network information

# Network name
NAME = "Cooper"
# size of the set on which the network operates, which is [-A, A]
A = 10  
# Number of points on which the network operates
N = 512
# Number of fourier modes 
MODES = 256
# Number of parameters for each lhidden layer in the network
HIDDEN_LAYERS = [1024, 2048, 1024]

### Training information

# Number of epochs used in training
EPOCH = 30
# Number of functions in a batch
BATCH_SIZE = 512
# Number of batches used in an epoch
NUM_BATCHES = 1000
# Training batch type. alternatives are "poly", "smooth" or "well"
TRAINING_TYPE = "mixed" 

### Testing information

# Number of functions in a testing batch
TESTING_BATCH_SIZE = 256
# Number of batches used in testing
TESTING_NUM_BATCHES = 200
# testing batch type. alternatives are "poly", "smooth", "well" or "mixed"
TESTING_TYPE = "mixed"
# Wether to plot some of the graphs to compare teoric and model phi during testing

# Wether to save testing results to a file
SAVE_RESULTS = True 