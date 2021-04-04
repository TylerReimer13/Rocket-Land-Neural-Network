import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim
import pandas as pd
from math import sin, cos
from animate import animated_plot
from sklearn.preprocessing import StandardScaler
import joblib
import sys

sys.modules['sklearn.externals.joblib'] = joblib


def dynamics_step(state, control_1, control_2, delt_t):
    x_, y_, thet_, m_, xdot_, ydot_, thetdot_ = state
    u1_ = control_1
    u2_ = control_2

    Iz = m_ * (3 * (r ** 2) + (h ** 2)) / 12.

    xddot = ((u1_ * thrust_mag) * sin(thet_ + u2_)) / m_
    yddot = (((u1_ * thrust_mag) * cos(thet_ + u2_)) / m_) - grav
    thetddot = (h * (u1_ * thrust_mag) * sin(u2_)) / (2 * Iz)

    xdot_ += xddot * delt_t
    ydot_ += yddot * delt_t
    thetdot_ += thetddot * delt_t

    x_ += xdot_ * delt_t
    y_ += ydot_ * delt_t
    thet_ += thetdot_ * delt_t
    m_ += -((u1_ * thrust_mag) / (g0 * Isp)) * delt_t

    return np.array([x_, y_, thet_, m_, xdot_, ydot_, thetdot_]).flatten()


def eval_model(test_x, u1_true=None, u2_true=None, path=None):
    x_sc, y_sc = load_scalers()

    x_test_orig = test_x.copy()
    test_x = np.array(test_x)
    scaled_test_x = x_sc.transform(test_x.reshape(1, -1))
    x_test = torch.tensor(scaled_test_x, dtype=torch.float32)

    trained = torch.load(path)
    trained.eval()
    scaled_pred = trained(x_test).detach().numpy()
    pred = y_sc.inverse_transform(scaled_pred)
    pred = pred.flatten()

    u1_pred = pred[:NU]
    u2_pred = pred[NU:]

    dt = x_test_orig[7]

    # x, y, thet, m, xdot, ydot, thetdot
    pred_state_vec = np.array([x_test_orig[0], x_test_orig[1], x_test_orig[2], x_test_orig[3], x_test_orig[4],
                               x_test_orig[5], x_test_orig[6]]).flatten()
    pred_states = pred_state_vec.copy()

    true_state_vec = np.array([x_test_orig[0], x_test_orig[1], x_test_orig[2], x_test_orig[3], x_test_orig[4],
                               x_test_orig[5], x_test_orig[6]]).flatten()
    true_state_vec_hist = true_state_vec.copy()

    for t in range(NU):
        pred_state_vec = dynamics_step(pred_state_vec, u1_pred[t], u2_pred[t], dt)
        pred_states = np.vstack((pred_states, pred_state_vec))

        if u1_true is not None and u2_true is not None:
            true_state_vec = dynamics_step(true_state_vec, u1_true[t], u2_true[t], dt)
            true_state_vec_hist = np.vstack((true_state_vec_hist, true_state_vec))

    plt.title('Rocket Trajectory')
    plt.plot(pred_states[0, 0], pred_states[0, 1], 'bX', label='Start')
    plt.plot(pred_states[:, 0], pred_states[:, 1], label='Predicted')
    plt.plot(pred_states[-1, 0], pred_states[-1, 1], 'rX', label='End')

    if u1_true is not None and u2_true is not None:
        plt.plot(true_state_vec_hist[:, 0], true_state_vec_hist[:, 1], 'g--', label='True')

    plt.legend()
    plt.xlabel('Downrange (m)')
    plt.ylabel('Altitude (m)')
    plt.grid()
    plt.savefig('results/Trajectory.png')
    plt.show()

    animated_plot(pred_states[:, 0], pred_states[:, 1], pred_states[:, 2], u1_pred, u2_pred, NU)


def save_scalers(scx, scy, path='models/sc_'):
    joblib.dump(scx, '{}x.bin'.format(path), compress=True)
    joblib.dump(scy, '{}y.bin'.format(path), compress=True)


def load_scalers(path='models/sc_'):
    scx = joblib.load('{}x.bin'.format(path))
    scy = joblib.load('{}y.bin'.format(path))
    return scx, scy


def scale_data(x_train, x_test, y_train, y_test):
    sc_x, sc_y = StandardScaler(), StandardScaler()

    x_train = sc_x.fit_transform(x_train)
    x_test = sc_x.transform(x_test)

    y_train = sc_y.fit_transform(y_train)
    y_test = sc_y.transform(y_test)

    save_scalers(sc_x, sc_y)

    return x_train, x_test, y_train, y_test


class LandNet(nn.Module):
    def __init__(self, in_dims, hidden, out_dims):
        super(LandNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dims, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dims))

        self.optimizer = optim.Adam(self.parameters(), lr=1e-3)
        self.loss_func = nn.MSELoss()
        self.val_loss_func = nn.MSELoss()

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

        self.loss_val = torch.tensor([np.inf])
        self.test_loss_val = torch.tensor([np.inf])

        self.epoch = 0

        self.train_losses = []
        self.test_losses = []

        self.val_freq = 5

    @property
    def val_loss(self):
        return round(self.test_loss_val.item(), 5)

    def forward(self, x):
        return self.net(x)

    def test_step(self, xtest, ytest):
        with torch.no_grad():
            pred = self(xtest)
            self.test_loss_val = self.val_loss_func(pred, ytest)

    def training_step(self, xtrain, ytrain, xtest, ytest):
        self.optimizer.zero_grad()
        y_pred = self(xtrain)
        self.loss_val = self.loss_func(y_pred, ytrain)
        self.loss_val.backward()
        self.optimizer.step()

        if self.epoch % self.val_freq == 0:
            self.test_step(xtest, ytest)
            print('EPOCH: ', self.epoch, 'TEST LOSS: ', self.val_loss)

        self.train_losses.append(self.loss_val.item())
        self.test_losses.append(self.val_loss)

        self.epoch += 1

    def save(self, path='nnet.pt'):
        torch.save(self, path)

    def plot_losses(self):
        plt.title('Train and Test Losses')
        start_at = 0
        plt.plot([epoch for epoch in range(self.epoch)[start_at:]], self.train_losses[start_at:], label='train')
        plt.plot([epoch for epoch in range(self.epoch)[start_at:]], self.test_losses[start_at:], label='test')
        plt.legend()
        plt.grid()
        plt.savefig('results/Neural Net Training Losses')
        plt.show()


def test_train_nnet(train=False):
    # 8169 total
    N_TRAIN = 7450  # Number of training samples

    # Read in initial states from excel, and remove unfeasible solutions
    init_data_raw = pd.read_excel('data/landing_data_init.xlsx')
    init_data_filtered = init_data_raw.loc[:, ~(init_data_raw == -np.inf).any()]
    init_data_filtered = init_data_filtered.to_numpy()[:, :]
    init_data_filtered = zip(*init_data_filtered)

    # Split x (input) data into train and test
    x = np.array(list(init_data_filtered))
    xtrain = x[:N_TRAIN, :8]
    xtest = x[N_TRAIN:, :8]

    # Read in optimal controls from excel, and remove unfeasible solutions
    control_data_raw = pd.read_excel('data/landing_data_control.xlsx')
    control_data_filtered = control_data_raw.loc[:, ~(control_data_raw == -np.inf).any()]
    control_data_filtered = control_data_filtered.to_numpy()
    control_data_filtered = zip(*control_data_filtered)

    # Split y (output) data into train and test
    y = np.array(list(control_data_filtered))
    ytrain = y[:N_TRAIN, :]
    ytest = y[N_TRAIN:, :]

    # Scale data so that each column has zero mean
    xtrain, xtest, ytrain, ytest = scale_data(xtrain, xtest, ytrain, ytest)

    # Convert data arrays to tensors
    xtrain_tensor = torch.tensor(xtrain, dtype=torch.float32)
    xtest_tensor = torch.tensor(xtest, dtype=torch.float32)
    ytrain_tensor = torch.tensor(ytrain, dtype=torch.float32)
    ytest_tensor = torch.tensor(ytest, dtype=torch.float32)

    # Select any set of initial conditions to plot results for
    test_idx = 8085
    init_x = x[test_idx, 0]
    init_y = x[test_idx, 1]
    init_thet = x[test_idx, 2]
    init_mass = x[test_idx, 3]
    init_vel_x = x[test_idx, 4]
    init_vel_y = x[test_idx, 5]
    init_thet_dot = x[test_idx, 6]
    init_dt = x[test_idx, 7]

    test_init = [float(init_x), float(init_y), float(init_thet), float(init_mass), float(init_vel_x), float(init_vel_y),
                 float(init_thet_dot), float(init_dt)]

    u1_true = y[test_idx, :NU]
    u2_true = y[test_idx, NU:]

    lander = LandNet(8, 175, 200)

    # Train neural net on training data
    if train:
        goal = .01  # .0124 MSE currently (dataset scaled) 3 layers, 175 neurons ea., ReLU and MSE loss
        n_epochs = 20_000  # Number of times to run the training loop

        for i in range(n_epochs):
            lander.training_step(xtrain_tensor, ytrain_tensor, xtest_tensor, ytest_tensor)

            if lander.val_loss <= goal:
                print('**************STOPPING EARLY**************')
                break

        lander.save(model_path)
        lander.plot_losses()

    # Show results of neural net on test data
    else:
        eval_model(test_init, u1_true, u2_true, model_path)


if __name__ == "__main__":
    """
    Running this file can train the neural network on the previously generated data, or do a validation pass on
    previously unseen data. To switch between the two, set the 'train' bool to True or False, respectively.
    """
    NU = 100
    grav = 9.81  # Earth gravity
    Isp = 100.
    g0 = 9.81
    h = 70  # Rocket height
    r = 3.7 / 2  # Rocket radius
    thrust_mag = 2000.

    model_path = 'models/nnet_v1.pt'  # nnet_v1 .01284 test loss, 3 layers, 175 neurons each

    test_train_nnet(train=False)

