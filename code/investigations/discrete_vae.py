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

def loss_func(x, y, mu, log_var, subsample_mode, weights):

  kld_weight = 1e-1

  if subsample_mode: 

    recons_loss = torch.nn.functional.mse_loss(x, y)
    kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

  else: 

    recons_loss = torch.sum(weights*torch.nn.functional.mse_loss(x, y, reduce=False))
    kld_loss = torch.sum(-0.5 * torch.sum(weights*(1 + log_var - mu ** 2 - log_var.exp()), dim = 1), dim = 0)

  loss = recons_loss + kld_weight * kld_loss

  return loss


if __name__ == '__main__':

  subsample_modes = [True,False]
  weights = np.array([0.1,0.3,0.5,0.1])
  values = np.arange(weights.shape[0])
  num_samples = 5000

  results = np.zeros((num_samples,len(subsample_modes)))

  # gen data
  choice_idxs = np.random.choice(values.shape[0],num_samples,p=weights)
  subsample_data = np.array([values[choice_idx] for choice_idx in choice_idxs])
  np.random.shuffle(subsample_data)  

  for i_mode, subsample_mode in enumerate(subsample_modes): 

    if subsample_mode:
      # subsample
      # choice_idxs = np.random.choice(values.shape[0],num_samples,p=weights)
      # x = np.array([values[choice_idx] for choice_idx in choice_idxs])
      # np.random.shuffle(x)
      x = subsample_data

    else: 
      # give as input the distribution weights 
      x = values 

    # train VAE
    learning_rate = 1e-3

    model = VAE()
    x_torch = torch.from_numpy(x).float().unsqueeze(1)
    weights_torch = torch.from_numpy(weights).float().unsqueeze(1)

    # loss_fn = torch.nn.MSELoss(reduction='sum')
    best_model = model 
    best_loss = np.inf 
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for t in range(10000):
        # Forward pass: compute predicted y by passing x to the model.
        x_pred, mu, log_var = model(x_torch)

        # Compute and print loss.
        loss = loss_func(x_pred, x_torch, mu, log_var, subsample_mode, weights_torch)
        if t % 100 == 99:
            print(t, loss.item())
            if loss.item() < best_loss:
              best_model = model

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
    results[:,i_mode] = best_model.decoder(z).detach().numpy().squeeze()

  # plot 
  fig,axs = plt.subplots(ncols=3)
  bins = np.arange(weights.shape[0]+1)
  axs[0].hist(subsample_data, bins=bins, density=True, alpha=0.5)
  axs[0].set_title('data')
  for i_mode, subsample_mode in enumerate(subsample_modes): 
    if subsample_mode:
      title = "subsampling"
    else:
      title = "weights"
    axs[1+i_mode].hist(results[:,i_mode], bins=bins, density=True, alpha=0.5)
    axs[1+i_mode].set_title(title)
  plt.savefig('temp.png')
  plt.show()
