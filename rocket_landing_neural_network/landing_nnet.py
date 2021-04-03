import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim
import pandas as pd
from math import sin, cos
from animate import animated_plot


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


def eval_model(path, x_test):
    trained = torch.load(path)
    trained.eval()
    pred = trained(x_test).detach().numpy()

    u1_pred = pred[:NU]
    u2_pred = pred[NU:]

    tf = test_init[2]
    dt = tf / NU

    # x, y, thet, m, xdot, ydot, thetdot
    pred_state_vec = np.array([x_test[0], x_test[1], 0., 100., 0., 0., 0.]).flatten()
    pred_states = pred_state_vec.copy()

    true_state_vec = np.array([x_test[0], x_test[1], 0., 100., 0., 0., 0.]).flatten()
    true_state_vec_hist = true_state_vec.copy()

    for t in range(NU):
        pred_state_vec = dynamics_step(pred_state_vec, u1_pred[t], u2_pred[t], dt)
        pred_states = np.vstack((pred_states, pred_state_vec))

        true_state_vec = dynamics_step(true_state_vec, u1_true[t], u2_true[t], dt)
        true_state_vec_hist = np.vstack((true_state_vec_hist, true_state_vec))

    plt.title('Rocket Trajectory')
    plt.plot(pred_states[0, 0], pred_states[0, 1], 'bX', label='Start')
    plt.plot(pred_states[:, 0], pred_states[:, 1], label='Predicted')
    plt.plot(pred_states[-1, 0], pred_states[-1, 1], 'rX', label='End')
    plt.plot(true_state_vec_hist[:, 0], true_state_vec_hist[:, 1], 'g--', label='True')
    plt.legend()
    plt.xlabel('Downrange (m)')
    plt.ylabel('Altitude (m)')
    plt.grid()
    plt.show()

    animated_plot(pred_states[:, 0], pred_states[:, 1], pred_states[:, 2], u1_pred, u2_pred, NU)


class LandNet(nn.Module):
    def __init__(self, in_dims, hidden, out_dims):
        super(LandNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dims, hidden),
            nn.Tanh(),
            nn.Linear(hidden, out_dims))

        self.optimizer = optim.Adam(self.parameters(), lr=5e-3)
        self.loss_func = nn.MSELoss()

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

        self.loss_val = torch.tensor([np.inf])
        self.test_loss_val = torch.tensor([np.inf])

        self.epoch = 0

        self.train_losses = []
        self.test_losses = []

    @property
    def val_loss(self):
        return round(self.test_loss_val.item(), 5)

    def forward(self, x):
        return self.net(x)

    def test_step(self):
        with torch.no_grad():
            pred = self(xtest_tensor)
            self.test_loss_val = self.loss_func(pred, ytest_tensor)

    def training_step(self):
        self.optimizer.zero_grad()
        y_pred = self(xtrain_tensor)
        self.loss_val = self.loss_func(y_pred, ytrain_tensor)
        self.loss_val.backward()
        self.optimizer.step()

        if self.epoch % validation_freq == 0:
            self.test_step()
            print('EPOCH: ', self.epoch, 'TEST LOSS: ', self.val_loss)

        self.train_losses.append(self.loss_val.item())
        self.test_losses.append(self.val_loss)

        self.epoch += 1

    def save(self, path='nnet.pt'):
        torch.save(self, path)

    def plot_losses(self):
        plt.title('Train and Test Losses')
        start_at = 50
        plt.plot([epoch for epoch in range(self.epoch)[start_at:]], self.train_losses[start_at:], label='train')
        plt.plot([epoch for epoch in range(self.epoch)[start_at:]], self.test_losses[start_at:], label='test')
        plt.legend()
        plt.grid()
        plt.show()


if __name__ == "__main__":
    # nnet: Test Loss = .0001, 1 layer, 50 neurons, 5e-3 lr, 2000 epochs
    # nnet_v2: Test Loss = .00008, 1 layer, 35 neurons, 5e-3 lr, 7500 epochs
    NU = 100
    grav = 9.81  # Earth gravity
    Isp = 100.
    g0 = 9.81
    h = 70  # Falcon 9 height
    r = 3.7 / 2  # Falcon 9 radius
    thrust_mag = 1000.

    lander = LandNet(3, 30, 200)

    N_TRAIN = 425  # Number of training samples
    validation_freq = 5  # Compute validation loss every n steps

    # Read in initial states from excel, and remove unfeasible solutions
    init_data_raw = pd.read_excel('data/landing_data_init.xlsx')
    init_data_filtered = init_data_raw.loc[:, ~(init_data_raw == -np.inf).any()]
    init_data_filtered = init_data_filtered.to_numpy()[:3, :]
    init_data_filtered = zip(*init_data_filtered)

    # Split x (input) data into train and test
    x = np.array(list(init_data_filtered))
    xtrain = x[:N_TRAIN, :]
    xtest = x[N_TRAIN:, :]

    xtrain_tensor = torch.tensor(xtrain, dtype=torch.float32)
    xtest_tensor = torch.tensor(xtest, dtype=torch.float32)

    # Read in optimal controls from excel, and remove unfeasible solutions
    control_data_raw = pd.read_excel('data/landing_data_control.xlsx')
    control_data_filtered = control_data_raw.loc[:, ~(control_data_raw == -np.inf).any()]
    control_data_filtered = control_data_filtered.to_numpy()
    control_data_filtered = zip(*control_data_filtered)

    # Split y (output) data into train and test
    y = np.array(list(control_data_filtered))
    ytrain = y[:N_TRAIN, :]
    ytest = y[N_TRAIN:, :]

    ytrain_tensor = torch.tensor(ytrain, dtype=torch.float32)
    ytest_tensor = torch.tensor(ytest, dtype=torch.float32)

    # Select any set of initial conditions to plot results for
    test_idx = 481
    init_x = x[test_idx, 0]
    init_y = x[test_idx, 1]
    init_T = x[test_idx, 2]
    test_init = [float(init_x), float(init_y), float(init_T)]

    u1_true = y[test_idx, :NU]
    u2_true = y[test_idx, NU:]

    model_path = 'models/nnet_v2.pt'
    train = False

    # Train neural net on training data
    if train:
        goal = .00008  # Validation loss below this value will trigger early stopping
        n_epochs = 7500  # Number of times to run the training loop

        for i in range(n_epochs):
            lander.training_step()

            if lander.val_loss <= goal:
                print('**************STOPPING EARLY**************')
                break

        lander.save(model_path)
        lander.plot_losses()

    # Show results of neural net on test data
    else:
        test = torch.tensor(test_init)
        eval_model(model_path, test)

