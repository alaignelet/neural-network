import numpy as np
import time
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from nn_lib import (
    MultiLayerNetwork,
    Trainer,
    Preprocessor,
    save_network,
    load_network,
)
from illustrate import illustrate_results_FM

# Set seed for reproducible results
torch.manual_seed(24)

# Load the data
#data = np.loadtxt('FM_dataset.dat')


def main():
    #######################################################################
    #                       ** START OF YOUR CODE **
    #######################################################################
    """"
    Main function, executed when executing this file in command line.
    
    Calls the predict_hidden(dataset) function which predicts the output of the 
    model for a hidden dataset. 
    One has to provide the path to the hidden dataset via the variable 
    "hidden_data_path".
    If such a path is not provided, the prediction will use a default dataset 
    (the training one).
    """

    print("########\n User information: the execution of this file predicts the labels of a hidden dataset that you should provide"
          "in the corresponding variable in the main.\n If you have NOT provided the path for your hidden dataset yet, this "
          "function will by default predict the labels of the training dataset.\n ########")

    default_data_path = 'FM_dataset.dat'

    # Please put the path to your hidden dataset here
    hidden_data_path = None

    if hidden_data_path is not None:
        dataset = np.loadtxt(hidden_data_path)
    else:
        dataset = np.loadtxt(default_data_path)
    predict_hidden(dataset)

    #######################################################################
    #                       ** END OF YOUR CODE **
    #######################################################################
    # illustrate_results_ROI(network, prep)


# <--------------- Neural Network Architecture ----------------> #

# Multi-target regression
class MultiRegression(nn.Module):
    '''
    Class for the neural network to perform a multi-target regression.
    
    Creates a neural network with 3 input neurons, 3 hidden layers, and 3
    output features (one for each coordinate). The ReLU function is the 
    activation function.
    '''
    
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(in_features = 3, out_features = 128) # Bias included by default
        self.lin2 = nn.Linear(in_features = 128, out_features = 64) 
        self.lin3 = nn.Linear(in_features = 64, out_features = 32) 
        self.lin4 = nn.Linear(in_features = 32, out_features = 3) # Position along 3 axes
        
    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = F.relu(self.lin3(x))
        x = self.lin4(x)
        return x

# <--------------- END Neural Network Architecture ----------------> #

# Separates the inputs and the labels, and normalises the inputs.
def data_preprocessing(data):
    '''
    Preprocesses the input data.
    
    The variables is centred (mean set to 0) and normalised (standard 
    deviation set to 1).
    Returns the input features and the labels separately.
    '''
    X, y = data[:, :3], data[:, 3:]
    mean, std = X.mean(0), X.std(0) # Mean and standard deviation for each variable
    inputs = (X - mean) / std
    return inputs, y


# Splits the data in three datasets
def split_data(dataset, valid_size = 0.2, test_size = 0.2):
    '''
    Splits the input data into three datasets.
        
    The output datasets are as following: the first one for training, 
    the next one for validation, and the last one for testing. 
    The output also converts the data in tensors and separates the features 
    from the label.
    '''
    # Normalisation of the inputs
    X, y = data_preprocessing(dataset)
    
    # Data split into training, validation and testing datasets
    valid_test_size = valid_size / (1 - test_size)
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size = test_size, 
                                                        random_state = 24)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, 
                                                      test_size = valid_test_size, 
                                                      random_state = 24)
    
    # Conversion into tensors
    X_train, X_val, X_test = torch.Tensor(X_train), torch.Tensor(X_val), torch.Tensor(X_test)
    y_train, y_val, y_test = torch.Tensor(y_train), torch.Tensor(y_val), torch.Tensor(y_test)
    return X_train, X_val, X_test, y_train, y_val, y_test
    

# Loads the data
def data_loader(dataset, batch_size, valid_size = 0.2, test_size = 0.2):
    '''
    Converts the input data into three datasets in a DataLoader object.
    
    Using the previous split function, converts the data into three DataLoader
    objects for training, validation and testing.
    '''
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(dataset, valid_size, test_size)
    
    # With TensorData and DataLoader
    train_data = TensorDataset(X_train, y_train)
    train_dl = DataLoader(train_data, batch_size = batch_size)
    val_data = TensorDataset(X_val, y_val)
    val_dl = DataLoader(val_data, batch_size = batch_size)
    test_data = TensorDataset(X_test, y_test)
    test_dl = DataLoader(test_data, batch_size = batch_size)
    return train_dl, val_dl, test_dl

    
def save_main_model():
    '''
    Former main function to solve the given problem.
    
    Trains a neural network on the training set with the best hyperparameters 
    given by the grid search in order to create a model. 
    Computes interesting metrics on the validation set and the testing set 
    with the final model. 
    The final model is saved for later usage possibility.
    '''
    dataset = np.loadtxt("FM_dataset.dat")
    #######################################################################
    #                       ** START OF YOUR CODE **
    #######################################################################
    # Set seed for reproducible results
    torch.manual_seed(24)

    # Variables
    epochs = 100  # Number of epochs
    batch_size = 25 #75 
    lr = 0.00005  # Learning rate #0.00015
    
    # Preprocessing and data loading
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(dataset)
    train_dl, val_dl, test_dl = data_loader(dataset, batch_size)

    # Create the neural network
    model = MultiRegression()
    opt = optim.SGD(model.parameters(), lr = lr, momentum = 0.4)
    loss_func = F.mse_loss # Mean squared error as a loss function
    
    # Save loss results per epoch for plots
    train_loss_history, val_loss_history = [], []
    
    # Train the neural network
    for epoch in range(epochs):
        # On the training dataset
        model.train()
        for X_batch, y_batch in train_dl:
    
            pred = model(X_batch)
            loss = loss_func(pred, y_batch)
    
            loss.backward()
            opt.step() # Updates the parameters thanks to the gradient
            opt.zero_grad() # Clears the gradients
        
        # Training log
        print("Train Epoch: {:02d} -- Loss: {:.4f}".format(#-- Batch: {:03d}
                        epoch, loss.item()))
        train_loss_history += [float(loss)]
        
        # On the validation dataset
        model.eval()
        with torch.no_grad():
            mae, mse, rmse, r2 = evaluate_architecture(model, X_val, y_val)
            #val_loss = mean_squared_error(y_val, model(X_val).detach().numpy())
        val_loss_history += [mse]

    
    # Metrics on the test dataset
    mae, mse, rmse, r2 = evaluate_architecture(model, X_test, y_test, True)
    
    # Save the best model
    torch.save(model.state_dict(), 'best_model_reg.pth')
    #######################################################################
    #                       ** END OF YOUR CODE **
    #######################################################################
    #illustrate_results_FM(network, prep)
    
def plot_loss_curves(train, val, cut_init = 5):
    '''
    Plots the training and validation loss curves.
    
    Using the list of losses during the training and the validation in the main
    function, this function plots the mean squared error for each epoch.
    '''
    epochs = len(train)
    epoch = [i for i in range(cut_init, epochs)]
    fig, ax = plt.subplots()
    ax.plot(epoch, train[cut_init:], label = 'train')
    ax.plot(epoch, val[cut_init:], label = 'validation')
    ax.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Mean squared error')
    plt.show()

#plot_loss_curves(train_loss_history, val_loss_history)

# Question 2.2
def evaluate_architecture(model, X, y, test_data = False):
    '''
    Prints and returns interesting metrics for a regression problem.
    
    The considered metrics are: the mean absolute error, the mean squared 
    error, the root mean squared error and the R2 score.
    '''
    with torch.no_grad():
        y_pred = model(X).detach().numpy()
    # Mean absolute error
    mae = mean_absolute_error(y, y_pred)
    # Mean squared error
    mse = mean_squared_error(y, y_pred)
    # Root mean squared error
    rmse = np.sqrt(mse)
    # R2 score
    r2 = r2_score(y, y_pred, multioutput = 'uniform_average')
    
    if test_data:
        print("\nTest set: Mean absolute error: {:.4f},  R2 score: {:.4f}".format(
                mae, r2))
        print("\t  Root mean squared error: {:.4f}\n".format(
                rmse))
    else:
        print("\nValid set: Mean absolute error: {:.4f}, R2 score: {:.4f}".format(
                mae, r2))
        print("\t  Root mean squared error: {:.4f}\n".format(
                rmse))
    return mae, mse, rmse, r2

#print('test error:', evaluate_architecture(model, X_test, y_test, True))


# Question 2.3
# Grid Search
def grid_search(epochs = 100, batch_size = 25):
    '''
    Performs a grid search to tune the learning rate and the momentum.
    
    For a given number of epochs and of batch_size, the function will return
    the best set of parameters and the best MSE.
    '''
    dataset = np.loadtxt("FM_dataset.dat")
    # Set seed for reproducible results
    torch.manual_seed(24)
    
    # Preprocessing and data loading
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(dataset)
    train_dl, val_dl, test_dl = data_loader(dataset, batch_size)
    
    # Grid search hyperparameters
    lr = [0.00005, 0.0001, 0.00015, 0.0002, 0.00025, 0.0003]
    momentum = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    parameters = [lr, momentum]

    best_mse = 10**6 # Initialisation
    best_params = [None, None]


    #time0 = time.time()
    for param1 in parameters[0]:
        print(param1)
        for param2 in parameters[1]:
            print(param2)
            # Create the neural network
            model = MultiRegression()
            opt = optim.SGD(model.parameters(), lr = param1, momentum = param2)
            loss_func = F.mse_loss # Mean squared error as a loss function
            
            # Train the neural network
            for epoch in range(epochs):
                # On the training dataset
                model.train()
                for X_batch, y_batch in train_dl:
            
                    pred = model(X_batch)
                    loss = loss_func(pred, y_batch)
            
                    loss.backward()
                    opt.step() # Updates the parameters thanks to the gradient
                    opt.zero_grad() # Clears the gradients
            
            #print(np.isnan(loss))
            if not np.isnan(loss.detach().numpy()): 
                # On the validation dataset
                model.eval()
                with torch.no_grad():
                    mae, mse, rmse, r2 = evaluate_architecture(model, X_val, y_val)
                
                if mse < best_mse:
                    best_mse = mse
                    best_params[0], best_params[1] = param1, param2
                
    return best_params, best_mse

#grid_search()
#([5e-05, 0.4], 15.671478)


def predict_hidden(dataset):
    '''
    Predicts the performance of the model on a hidden dataset.
    
    Preprocesses the hidden dataset, loads the best performing model and 
    computes the predicted output. The predicted output is a Numpy array of 
    shape (n_samples, 3).
    '''
    # -------  Data loaders -------- #
    X, y = data_preprocessing(dataset)
    X = torch.Tensor(X)

    # ------- Instantiate model ----- #
    model = MultiRegression()

    # ----- Load our best model ------ #
    model.load_state_dict(torch.load('best_model_reg.pth'))

    # ----- Compute the output ------ #
    with torch.no_grad():
        y_pred = model(X)

    return y_pred.detach().numpy() 

# <------------ Executes the main function in command line -------------> #

if __name__ == "__main__":
    main()

# <-------------------------------- END --------------------------------> #

