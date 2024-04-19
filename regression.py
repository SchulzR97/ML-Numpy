import numpy as np
import layer
import activation as a
import matplotlib.pyplot as plt
import net
import data
import utils
import loss
import os

def plot_loss(loss_train, loss_val):
    plt.plot(loss_train, label='train')
    plt.plot(loss_val, label='val')
    plt.grid(which='both')
    plt.xlim(0, len(loss_train)-1)
    plt.minorticks_on()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig('loss.png')
    plt.close()

def plot_data(X, Y, T,):
    sorted_i = np.argsort(X, axis=0)[:,0]

    X = X[sorted_i]
    Y = Y[sorted_i]
    T = T[sorted_i]

    baselineX = np.array([np.min(X), np.max(X)])
    baselineY = np.array([0, 0])

    plt.title('Data')
    plt.minorticks_on()
    plt.grid(which='both')
    plt.scatter(X, T, label='target')
    plt.scatter(X, Y, label='prediction')
    plt.plot(X, Y-T, color='gray', label='difference')
    plt.plot(baselineX, baselineY, color='gray')
    plt.xlim(np.min(X), np.max(X))
    plt.xlabel('X')
    plt.ylabel('data')
    plt.legend()

    plt.savefig('data.png')
    plt.close()

# hyper parameters
N = 1000
MAX_NOISE = 0.05
EPOCHS = 5000000
LEARNING_RATE = 1e-3#1e-8
BATCH_SIZE = 32

# additional parameters
FILENAME = 'model.pkl'
LOAD_MODEL = False
PLOT_EPOCHS = 1000
MAVG_EPOCHS = 8000
WARM_UP_EPOCHS = 12000

if __name__ == '__main__':
    model = net.NeuralNetwork([
        layer.DenseLayer(1, 40),
        a.LeakyReLU(),

        layer.DenseLayer(40, 60),
        a.LeakyReLU(),

        layer.DenseLayer(60, 40),
        a.LeakyReLU(),

        layer.DenseLayer(40, 1)
    ])

    criterion = loss.MeanSquareError(reduction='sum')

    noise = -0.5 * MAX_NOISE + np.random.random((N, 1)) * MAX_NOISE

    X = np.linspace(-1., 1., N)
    X = X.reshape((N, 1))
    T = 0.5 + 0.5 * np.sin(np.pi * X) + noise

    X_train, T_train, X_val, T_val = data.split(X, T, train_prop=0.8, shuffle=True)

    ds_train = data.Dataset(X_train, T_train)
    ds_val = data.Dataset(X_val, T_val)

    losses_train = []
    losses_val = []
    losses_train_mavg = []
    losses_val_mavg = []

    if LOAD_MODEL and FILENAME is not None and os.path.isfile(FILENAME):
        model = net.load(FILENAME)

    epoch = 0
    while epoch < EPOCHS:
        # train
        model.train()
        batch_X, batch_T = ds_train.sample(BATCH_SIZE)
        batch_Y = model(batch_X)
        loss_train = criterion(batch_Y, batch_T)
        loss_train_value = criterion.value()
        model.backward(loss_train, LEARNING_RATE)

        # eval
        model.eval()
        batch_X, batch_T = ds_val.sample(BATCH_SIZE)
        batch_Y = model(batch_X)
        loss_val = criterion(batch_Y, batch_T)
        loss_val_value = criterion.value()

        losses_train.append(loss_train_value)
        losses_val.append(loss_val_value)

        if epoch > WARM_UP_EPOCHS + MAVG_EPOCHS:
            losses_train_mavg.append(np.average(losses_train[-MAVG_EPOCHS:]))
            losses_val_mavg.append(np.average(losses_val[-MAVG_EPOCHS:]))
            print(f'Epoch {epoch}/{EPOCHS} Loss_train {losses_train_mavg[-1]:0.8f} Loss_val {losses_val_mavg[-1]:0.8f}')
        else:
            losses_train_mavg.append(np.nan)
            losses_val_mavg.append(np.nan)
            print(f'Epoch {epoch}/{EPOCHS} Loss_train {np.average(losses_train[-int(0.2*epoch):]):0.8f} Loss_val {np.average(losses_val[-int(0.2*epoch):]):0.8f}')        

        if epoch%PLOT_EPOCHS == 0:
            if epoch >= WARM_UP_EPOCHS + MAVG_EPOCHS:
                plot_loss(losses_train_mavg, losses_val_mavg)
            else:
                plot_loss(utils.moving_average(losses_train, int(0.2*epoch)), utils.moving_average(losses_val, int(0.2*epoch)))
            model.eval()
            batch_X, batch_T = ds_val.sample(100)
            batch_Y = model(batch_X)
            plot_data(batch_X, batch_Y, batch_T)

            model.save(FILENAME)

        epoch += 1

    pass

    model.save(FILENAME)

    batch_X, batch_T = ds_val.sample(BATCH_SIZE)
    batch_Y = model(batch_X)
    plt.scatter(batch_X, batch_T, label='target')
    plt.scatter(batch_X, batch_Y, label='prediction')
    plt.legend()
    plt.show()

    plt.plot(utils.moving_average(losses_train, period=10000))
    plt.show()