import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

Version = 'GD' ## could be 'GD', 'MBGD', 'SGD'
Epoch = 100

class Model():
    def __init__(self, LR=0.0001):
        self.weight = np.random.randn(1, 2)
        self.lr = LR

    ## compute model output
    def forward(self, x):
        return self.weight.dot(x)
    
    ## MSE loss
    def Loss(self, y_hat, y):
        return ((y_hat-y)**2).sum()

    ## compute gradient of loss
    def d_Loss(self, x, y_hat, y):
        return (2*(y_hat-y)*x).sum(axis=1)

    def train(self, x, y):
        ## compute model output
        y_hat = self.forward(x).squeeze()

        ## compute loss and loss gradient
        loss = self.Loss(y_hat, y)
        d_loss = self.d_Loss(x, y_hat, y)

        ## update weight
        self.weight -= self.lr*d_loss
        return y_hat, loss

    def test(self, x, y):
        ## compute model output
        y_hat = self.forward(x).squeeze()
        ## compute loss
        loss = self.Loss(y_hat, y)
        return y_hat, loss

## read training and testing data
def read_data(train_data_pth = './train_data.csv', test_data_pth = './test_data.csv'):
    ## read csv file, `.sample` is for random shuffle
    train_data = pd.read_csv(train_data_pth)
    test_data = pd.read_csv(test_data_pth)

    ## extract train and test (x, y)
    ## `.T` is for data shape, to let it do dot product successfully
    train_x = train_data['x_train'].values
    train_x = np.expand_dims(train_x, axis=1)
    train_x = np.concatenate((train_x, np.ones((len(train_x), 1))), axis=1).T
    train_y = train_data['y_train'].values

    test_x = test_data['x_test'].values
    test_x = np.expand_dims(test_x, axis=1)
    test_x = np.concatenate((test_x, np.ones((len(test_x), 1))), axis=1).T
    test_y = test_data['y_test'].values

    return train_x, train_y, test_x, test_y

## plot the figure and save it
def plot_fig(train_x, train_y, test_x, test_y, train_loss_log, test_loss_log):
    ## plot model predict values in training data
    y_hat = model.test(train_x, train_y)
    plt.plot(train_x[0], y_hat[0])
    plt.scatter(train_x[0], train_y)
    plt.xlabel('X axis')
    plt.ylabel('Y axis')
    plt.title(f'{Version} Training Model Plot')
    plt.savefig(f'{Version}_train_model_plot.png')

    ## plot model predict values in testing data
    plt.close()
    y_hat = model.test(test_x, test_y)
    plt.plot(test_x[0], y_hat[0])
    plt.scatter(test_x[0], test_y)
    plt.xlabel('X axis')
    plt.ylabel('Y axis')
    plt.title(f'{Version} Testing Model Plot')
    plt.savefig(f'{Version}_test_model_plot.png')

    ## plot training loss curve
    plt.close()
    plt.plot(train_loss_log)
    plt.xlabel('Iter')
    plt.ylabel('Loss')
    plt.title(f'{Version} Training Loss Curve')
    plt.savefig(f'{Version}_train_loss.png')

    ## plot testing loss curve
    plt.close()
    plt.plot(test_loss_log)
    plt.xlabel('Iter')
    plt.ylabel('Loss')
    plt.title(f'{Version} Testing Loss Curve')
    plt.savefig(f'{Version}_test_loss.png')

if __name__ == "__main__":
    ## read data and declare model
    train_x, train_y, test_x, test_y = read_data()
    model = Model()

    ## two loss record list for displaying loss curve
    train_loss_log = []
    test_loss_log = []

    ## check the version of optimize method
    if Version == 'GD':
        BS = len(train_y)
    elif Version == 'MBGD':
        BS = 64
    elif Version == 'SGD':
        BS = 1
    else:
        print("Version Error!")
        exit(1)

    
    train_len = int(len(train_y)/BS)    # the times it need to do within an iteration
    for iter in range(Epoch):
        train_loss = 0
        ## iterate all the data in the input size of BS
        for idx in range(0, train_len):
            _, sub_train_loss = model.train(\
                train_x[:, idx*BS:(idx+1)*BS], \
                train_y[idx*BS:(idx+1)*BS]
            )
            train_loss += sub_train_loss

        ## if there's still remaining data the model hasn't compute, compute it
        if train_len*BS != len(train_y):
            _, sub_train_loss = model.train(\
                train_x[:, idx*BS:(idx+1)*BS], \
                train_y[idx*BS:(idx+1)*BS]
            )
        train_loss += sub_train_loss

        ## for testing data, we could feed all the data at one time since we don't need to optimize the weight
        test_y_hat, test_loss = model.test(test_x, test_y)

        ## compute the average loss and append to loss record list
        train_loss /= len(train_y)
        test_loss /= len(test_y)
        train_loss_log.append(train_loss)
        test_loss_log.append(test_loss)
    
    print(f'Model Weight and Intercepts: {model.weight}')
    print(f'Training loss {train_loss_log[-1]}, Testing loss {test_loss_log[-1]}')

    ## plot the figure and save it
    plot_fig(train_x, train_y, test_x, test_y, train_loss_log, test_loss_log)