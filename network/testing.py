import torch
import network.network as network
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import network.config as config

def testNetwork(device, statedict_path, testing_path, nplots):

    # Loads the model
    model = network.FourierNet().to(device)
    if(Path(statedict_path).exists()):
        print("Loading model...")
        state_dict = torch.load(statedict_path)
        model.load_state_dict(state_dict)
        model.eval()
        print("Model loaded correctly")
    else:
        raise Exception(f"File {statedict_path} doesn't exist. Can't train network")

    # Generates the testing set if it doesn't exist
    if(Path(testing_path).exists() == False):
        raise Exception(f"File {testing_path} doesn't exist. Can't train network")

    # Loads the testing set
    print(f"Loading testingset {testing_path}")
    testset = torch.load( testing_path, map_location="cpu")
    print("Testing set loaded correctly")

    # Checks if the file does not match the config file
    if(testset["function"].shape != (config.TESTING_NUM_BATCHES, config.TESTING_BATCH_SIZE, config.N)
        or testset["phi"].shape != (config.TESTING_NUM_BATCHES, config.TESTING_BATCH_SIZE, config.N)
        or testset["E"].shape != (config.TESTING_NUM_BATCHES, config.TESTING_BATCH_SIZE, 1)):

        raise Exception("The testing set found in ", testing_path, " does not match the config file.")

    # obtains the data from the testing set
    func_set = testset["function"]
    phi_set = testset["phi"]
    E_set = testset["E"]

    # Begins testing
    func_score = 0
    E_score = 0
    for i in range(config.TESTING_NUM_BATCHES):

        # Takes the current batch
        f = func_set[i].to(device, non_blocking=True)
        phi = phi_set[i].to(device, non_blocking=True)
        E = E_set[i].to(device, non_blocking=True)

        # Calculates the model phi and E
        m_E, m_phi, _ = model(f)

        # Function score remains low if the model phi^2 is similar to phi^2
        func_score += torch.mean( (m_phi**2 - phi**2)**2 )
        E_score += torch.mean( (E - m_E)**2 )

        # Prints progress
        print(f"\rTesting network: {i+1}/{config.TESTING_NUM_BATCHES} ({100*(i+1)/config.TESTING_NUM_BATCHES:.1f}%)", end="", flush=True)

        # Makes a graph N_GRAPHS times
        if(i % (config.TESTING_NUM_BATCHES // nplots) == 0):
            # Choses which function to plot
            k = torch.randint(0, config.TESTING_BATCH_SIZE, (1,))

            # takes the functions to plot from the batch
            f_plot = f[k].squeeze().detach().cpu().numpy()
            phi_plot = (phi[k]**2).squeeze().detach().cpu().numpy()
            m_phi_plot = (m_phi[k]**2).squeeze().detach().cpu().numpy()

            t = np.linspace(-config.A, config.A, config.N)

            # Plots the graph
            plt.plot(t, f_plot, label="V")
            plt.plot(t, phi_plot, label="Phi^2 (fd)")
            plt.plot(t, m_phi_plot, label="Phi^2 (Model)")
            plt.legend()

            # Prints energy info
            print()
            print(f"The plotted graph has fd energy {E[k].squeeze().item():.6f}, while model energy is {m_E[k].squeeze().item():.6f}")
            print("Close the graph to continue testing")

            plt.show()
    print()


    # Takes the average between all batches
    func_score /= config.TESTING_NUM_BATCHES
    E_score /= config.TESTING_NUM_BATCHES

    print(f"Testing done successfully. scored: func_score: {func_score}, E_score: {E_score}")