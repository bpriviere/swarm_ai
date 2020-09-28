import numpy as np
import matplotlib.pyplot as plt
import torch


class VAE(torch.nn.Module):
  def __init__(self):
      super(VAE, self).__init__()

      self.K = 1

      # input: x, output: u and sigma
      self.encoder = torch.nn.Sequential(
        torch.nn.Linear(1, 16),
        torch.nn.ReLU(),
        torch.nn.Linear(16, 16),
        torch.nn.ReLU(),
        torch.nn.Linear(16, 2*self.K),
      )

      # input: sample from latent space N(0,1), output: sample from X
      self.decoder = torch.nn.Sequential(
        torch.nn.Linear(self.K, 16),
        torch.nn.ReLU(),
        torch.nn.Linear(16, 16),
        torch.nn.ReLU(),
        torch.nn.Linear(16, 1),
      )

  def forward(self, x):

    # compute mu and log_var
    enc = self.encoder(x)
    mu = enc[:,0:self.K]
    log_var = enc[:,self.K:]

    # reparameterize
    std = torch.exp(0.5 * log_var)
    eps = torch.randn_like(std)
    z = mu + eps * std

    # decode
    dec = self.decoder(z)
    return dec, mu, log_var

def loss_func(x, y, mu, log_var):
  kld_weight = 1e-0
  recons_loss = torch.nn.functional.mse_loss(x, y)
  kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
  loss = recons_loss + kld_weight * kld_loss

  return loss


if __name__ == '__main__':

  # generate data
  num_samples = 10000
  x = np.empty((num_samples, 1))
  x[0:num_samples//2] = np.random.normal(0, 1, (num_samples//2, 1))
  x[num_samples//2:]  = np.random.normal(5, 0.5, (num_samples//2, 1))
  np.random.shuffle(x)

  # train VAE
  learning_rate = 1e-3

  model = VAE()
  x_torch = torch.from_numpy(x).float()

  # loss_fn = torch.nn.MSELoss(reduction='sum')
  optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
  for t in range(5000):
      # Forward pass: compute predicted y by passing x to the model.
      x_pred, mu, log_var = model(x_torch)

      # Compute and print loss.
      loss = loss_func(x_pred, x_torch, mu, log_var)
      if t % 100 == 99:
          print(t, loss.item())

      # Before the backward pass, use the optimizer object to zero all of the
      # gradients for the variables it will update (which are the learnable
      # weights of the model). This is because by default, gradients are
      # accumulated in buffers( i.e, not overwritten) whenever .backward()
      # is called. Checkout docs of torch.autograd.backward for more details.
      optimizer.zero_grad()

      # Backward pass: compute gradient of the loss with respect to model
      # parameters
      loss.backward()

      # Calling the step function on an Optimizer makes an update to its
      # parameters
      optimizer.step()


  # generate data using learned model
  z = torch.randn((num_samples, model.K))
  x_learned = model.decoder(z).detach().numpy()

  plt.hist(x, bins=30, density=True, label="training data", alpha=0.5)
  plt.hist(x_learned, bins=30, density=True, label="learned prediction", alpha=0.5)
  plt.legend()
  plt.show()
