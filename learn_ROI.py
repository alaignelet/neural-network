import numpy as np
import time
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.datasets as datasets

from sklearn.metrics import confusion_matrix
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import GridSearchCV



from nn_lib import (
    MultiLayerNetwork,
    Trainer,
    Preprocessor,
    save_network,
    load_network,
)
from illustrate import illustrate_results_ROI

def main():
    #######################################################################
    #                       ** START OF YOUR CODE **
    #######################################################################
    """"
    Main function, executed when executing this file in command line.
    Calls the predict_hidden(dataset) function which predicts the output of the model
    for a hidden dataset. One has to provide the path to the hidden dataset via the variable "hidden_data_path".
    If such a path is not provided, the prediction will use a default dataset (the training one).
    """

    print("########\n User information : the execution of this file predicts the labels of a hidden dataset that you should provide"
          "in the corresponding variable in the main.\n If you have NOT provided the path for your hidden dataset yet, this "
          "function will by default predicts the labels of the training dataset.\n ########")

    default_data_path = "ROI_dataset.dat"

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




# <--------------- Neural Networks Architectures ----------------> #

class OneHiddenLayerClassifier(nn.Module):
    """
    First model architecture class : one hidden layer fully connected neural network
    """
    # Define entities containing model weights in the constructor.
    def __init__(self, n_hidden):
        super().__init__()
        self.linear1 = nn.Linear(
            in_features=3, out_features=n_hidden, bias=True
        )
        self.linear2 = nn.Linear(
            in_features=n_hidden, out_features=4, bias=True
        )

    def forward(self, inputs):
        h = self.linear1(inputs)
        h = F.relu(h)
        h = self.linear2(h)
        return F.log_softmax(h, dim=1)

class ThreeHiddenLayerClassifier(nn.Module):
    """
        Second model architecture class : three hidden layers fully connected neural network
    """
    # Define entities containing model weights in the constructor.
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(
            in_features=3, out_features=256, bias=True
        )
        self.linear2 = nn.Linear(
            in_features=256, out_features=128, bias=True
        )
        self.linear3 = nn.Linear(
            in_features=128, out_features=64, bias=True
        )
        self.linear4 = nn.Linear(
            in_features=64, out_features=32, bias=True
        )
        self.linear5 = nn.Linear(
            in_features=32, out_features=4, bias=True
        )

    def forward(self, inputs):
        h = self.linear1(inputs)
        h = F.relu(h)

        h = self.linear2(h)
        h = F.relu(h)

        h = self.linear3(h)
        h = F.relu(h)

        h = self.linear4(h)
        h = F.relu(h)

        h = self.linear5(h)
        return F.log_softmax(h, dim=1)

# <--------------- END Neural Networks Architectures ----------------> #


# <--------------- Training and evaluation function for the Neural Networks ----------------> #

def train(model, train_loader, optimizer, epoch, log_interval=100, log=True):
    """
    A utility function that performs a basic training loop.

    For each batch in the training set, fetched using `train_loader`:
        - Zeroes the gradient used by `optimizer`
        - Performs forward pass through `model` on the given batch
        - Computes loss on batch
        - Performs backward pass
        - `optimizer` updates model parameters using computed gradient

    Prints the training loss on the current batch every `log_interval` batches.
    """
    loss_history = []
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        # Zeroes the gradient used by `optimizer`;
        optimizer.zero_grad()

        # Performs forward pass through `model` on the given batch;
        outputs = model(inputs)

        # Computes loss on batch; cross_entropy_one_hot is a custom wrapper function to deal with one-hot encoding
        loss = cross_entropy_one_hot(outputs, targets)

        # For logging purposes
        loss_history.append(float(loss))

        # Performs backward pass; steps backward through the computation graph,
        # computing the gradient of the loss wrt model parameters.
        loss.backward()

        # `optimizer` updates model parameters using computed gradient.
        optimizer.step()

        # Prints the training loss on the current batch every `log_interval`
        # batches.
        if log:
            if batch_idx % log_interval == 0:
                print(
                    "Train Epoch: {:02d} -- Batch: {:03d} -- Loss: {:.4f}".format(
                        epoch,
                        batch_idx,
                        # Calling `loss.item()` returns the scalar loss as a Python
                        # number.
                        loss.item(),
                    )
                )
    return np.array(loss_history).mean()/16


def evaluate_architecture(model, test_loader, test_data=False):
    """
    A utility function to compute the loss, accuracy, confusion matrix, recall and precision
    on a test set by iterating through the test set using the provided `test_loader`.
    """
    test_loss = 0.0
    correct = 0
    test_size = 0.
    ground_truth = np.array([])
    predictions = np.array([])

    with torch.no_grad():
        for inputs, targets in test_loader:
            test_size += inputs.shape[0]
            outputs = model(inputs)

            test_loss += cross_entropy_one_hot(outputs, targets).item()
            pred = outputs.argmax(dim=1, keepdim=True)
            targets = targets.argmax(dim=1, keepdim=True)

            ground_truth = np.append(ground_truth, np.array(targets))
            predictions = np.append(predictions, np.array(pred))

            correct += pred.eq(targets).sum().item()
    test_loss /= test_size
    correct /= test_size

    if test_data:
        print(
            "\nTest set: Average loss: {:.4f}, Accuracy: {:.4f}\n".format(
                test_loss, correct
            )
        )
    else:
        print(
            "\nValid set: Average loss: {:.4f}, Accuracy: {:.4f}\n".format(
                test_loss, correct
            )
        )
    cm = confusion_matrix(ground_truth, predictions)
    recall, precision = recall_precision(cm)

    return correct, cm, recall, precision, test_loss

def train_save_model():
    """
    Used to be the main function. Instantiates a model, trains it with the provided hyper parameters,
    evaluates it and saves it. This function has been used to save our best model
    """
    data_path = "ROI_dataset.dat"

    # Parameters
    EPOCH_NUMBER = 100
    batch_size = 16

    # First grid search
    # best_params = [0.05, 0.5]
    # Second grid search
    best_params = [0.08, 0.45]

    # Load the dataset
    dataset = np.loadtxt(data_path)

    # Data Loaders instantiation
    train_loader, valid_loader, test_loader = data_loader(dataset, batch_size, random_seed=1, valid_size=0.1, test_size=0.1)

    # Create instance of our model
    #model = OneHiddenLayerClassifier(n_hidden=256)
    model = ThreeHiddenLayerClassifier()

    # Create instance of optimizer
    optimizer = optim.SGD(model.parameters(), lr=best_params[0], momentum=best_params[1])

    # For plot purposes
    train_loss_history = []
    val_loss_history = []

    for epoch in range(EPOCH_NUMBER):
        train_loss = train(model, train_loader, optimizer, epoch)
        _, _, _, _, val_loss = evaluate_architecture(model, valid_loader)

        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)

    acc, confusion, recall, precision, test_loss = evaluate_architecture(model, test_loader, test_data=True)
    print("Confusion matrix on test set : ")
    print(confusion)

    print("Recall : ")
    print(recall)

    print("Precision : ")
    print(precision)

    torch.save(model.state_dict(), 'best_model.pth')

    plot_loss_curves(train_loss_history, val_loss_history)

    best_params, acc = grid_search()

    print("Best params ", best_params)
    print("Achieved accuracy of ", acc)


def grid_search():
    """
    Function used to perform grid search on learning rate and momentum
    :return:
    """
    EPOCH_NUMBER = 20
    batch_size = 16

    # First grid search
    # lr = [0.001, 0.005, 0.01, 0.05, 0.1]
    # momentum = [0.1, 0.3, 0.5, 0.7, 0.9]

    # Second grid search
    lr = [0.03, 0.04, 0.05, 0.06, 0.07, 0.08]
    momentum = [0.4, 0.45, 0.5, 0.55, 0.6]
    parameters = [lr, momentum]

    nb_combinations = len(parameters[0]) * len(parameters[1])
    to_be_done = "--"
    done = "**"
    counter = 0

    best_acc = 0
    best_params = [None, None]

    print("Status of grid search :     ", to_be_done * (nb_combinations - counter))

    time0 = time.time()
    for param1 in parameters[0]:
        for param2 in parameters[1]:
            model = ThreeHiddenLayerClassifier()
            # Create instance of optimizer
            optimizer = optim.SGD(model.parameters(), lr=param1, momentum=param2)

            list_acc = np.zeros(5)

            for seed in range(5):
                train_loader, valid_loader, test_loader = data_loader(data_path, batch_size, random_seed=seed)
                for epoch in range(EPOCH_NUMBER):
                    train(model, train_loader, optimizer, epoch, log=False)
                acc, _, _, _ = evaluate_architecture(model, valid_loader)
                list_acc[seed] = acc

            average_acc = list_acc.mean()
            if average_acc > best_acc:
                best_acc = average_acc
                best_params[0], best_params[1] = param1, param2

            counter += 1

            if counter == 1:
                time1 = time.time()
                duration = time1 - time0
            print("Status of grid search :     ", done * (counter) + to_be_done * (nb_combinations - counter))
            print("Estimated remaining time : ", round(duration * (nb_combinations - counter) / 60), " min")

    return best_params, best_acc

# <--------------- END Training and evaluation function for the Neural Networks ----------------> #


# <--------------- Data management ----------------> #

def data_loader(dataset, batch_size, random_seed, valid_size=0.1, test_size=0.1):
    """
    Creates training, validation and testing pytorch dataloaders to be used for training.
    :param dataset: raw dataset in ndarray format
    :param batch_size: batch_size
    :param random_seed: for reproducible results
    :param valid_size: percentage of dataset to use for validation purposes
    :param test_size: percentage of dataset to use for testing purposes
    :return: 3 objects of type pytorch DataLoaders, which can be used for training
    """
    inputs_, labels_ = data_preprocessing(dataset)
    #inputs_ = torch.Tensor(dataset[:,:3])
    #labels_ = torch.Tensor(dataset[:,3:]).long()
    dataset_ = torch.utils.data.TensorDataset(inputs_, labels_)

    # Split to train / valid / test dataset
    size_dataset = dataset.shape[0]
    indices = list(range(size_dataset))
    valid_split = int(np.floor(valid_size * size_dataset))
    test_split = int(np.floor(test_size * size_dataset))

    np.random.seed(random_seed)
    np.random.shuffle(indices)

    train_valid_idx, test_idx = indices[test_split:], indices[:test_split]
    train_idx, valid_idx = train_valid_idx[valid_split:], train_valid_idx[:valid_split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
    test_sampler = SubsetRandomSampler(test_idx)

    train_loader = torch.utils.data.DataLoader(
        dataset_, batch_size=batch_size, sampler=train_sampler)
    valid_loader = torch.utils.data.DataLoader(
        dataset_, batch_size=batch_size, sampler=valid_sampler)
    test_loader = torch.utils.data.DataLoader(
        dataset_, batch_size=batch_size, sampler=test_sampler)

    return train_loader, valid_loader, test_loader

def data_preprocessing(dataset):
    """
    Small data preprocessing performed : centers the data around zero and normalize it
    :param dataset: raw numpy dataset
    :return: normalized inputs with associated labels
    """
    # We first compute the mean and the standard deviation of dataset to normalise it.
    mean, std = dataset.mean(0)[:3], dataset.std(0)[:3]

    inputs = torch.Tensor((dataset[:, :3] - mean) / std)
    labels = torch.Tensor(dataset[:, 3:]).long()
    return inputs, labels

# <--------------- END Data management ----------------> #


# <--------------- Utils functions : metrics ----------------> #

def cross_entropy_one_hot(input, target):
    """
    Wrapper function to deal with one hot encoding
    :param input: prediction of the model
    :param target: ground truth labels, provided with one hot encoding
    :return: Cross entropy loss between the inputs
    """
    _, labels = target.max(dim=1)
    return nn.CrossEntropyLoss()(input, labels)


def recall_precision(cm):
    """
    Computes the corresponding recall and precision for each class
    :param cm: confusion matrix of the results. Can be obtained with the confusion_matrix function for the sklearn library
    :return: recall and precision of the classes
    """
    nb_classes = cm.shape[0]
    recall = np.zeros(nb_classes)
    precision = np.zeros(nb_classes)
    for idx in range(nb_classes):
        if (idx + 1) < nb_classes:
            false_negative = np.concatenate((cm[idx, :idx], cm[idx, (idx + 1):]))
            false_positive = np.concatenate((cm[:idx, idx], cm[(idx + 1):, idx]))
        else:
            false_negative = cm[idx, :idx]
            false_positive = cm[:idx, idx]
        true_positive = cm[idx, idx]
        recall[idx] = true_positive / (true_positive + false_negative.sum())
        precision[idx] = true_positive / (true_positive + false_positive.sum())
    return recall, precision

def metrics(prediction, target):
    """
    Main function to get metrics from a model
    :param prediction: predictions of the model
    :param target: ground truth label values
    :return: accuracy, confusion matrix, recall and precision
    """
    # Number of correct predictions
    correct = prediction.eq(target).sum().item()
    accuracy = correct / target.shape[0]

    # Confusion matrix
    cm = confusion_matrix(target, np.array(prediction))

    # Recall and precision
    recall, precision = recall_precision(cm)

    return accuracy, cm, recall, precision

def plot_loss_curves(train, val):
    """
    Plots the training and validation loss curves
    :param train: array of training losses over N epochs
    :param val: array of validation losses over N epochs
    :return: nothing, plots the curves.
    """
    NB_EPOCH = len(train)
    epoch = [i for i in range(1, NB_EPOCH + 1)]
    plt.scatter(epoch, train)
    plt.scatter(epoch, val)
    plt.show()

# <--------------- END Utils functions : metrics ----------------> #


# <--------------- Testing function on hidden datasets ----------------> #


def predict_hidden(dataset):
    """
    Performs label prediction on a raw hidden dataset by using a pretrained/saved best model
    :param dataset: raw hidden dataset on ndarray format
    :return: the model label predictions
    """
    # -------  Data loaders -------- #
    inputs, labels = data_preprocessing(dataset)

    # ------- Instantiate model ----- #
    model = ThreeHiddenLayerClassifier()

    # ----- Load our best model ------ #
    model.load_state_dict(torch.load('best_model.pth'))

    # ----- Compute the output ------ #
    output = model(inputs)

    # ----- Squeeze pred and target from one hot to int representation ------ #
    prediction = output.argmax(dim=1, keepdim=True)
    target = labels.argmax(dim=1, keepdim=True)

    # ----- Compute the metrics of our test ------ #
    accuracy, cm, recall, precision = metrics(prediction, target)

    # ----- Print the results ------ #
    print("Accuracy on test set : ", accuracy)
    print("Confusion matrix :")
    print(cm)
    print("Recall")
    print(recall)
    print("Precision")
    print(precision)

    return prediction

# <--------------- END Testing function on hidden datasets ----------------> #


# <--------------- Executes the main function in command line ----------------> #

if __name__ == "__main__":
    main()

# <------------------------------------- END -----------------------------------> #
