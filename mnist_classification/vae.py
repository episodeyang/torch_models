from params_proto import cli_parse
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm


@cli_parse
class Args:
    batch_size = 100
    num_workers = 5
    pin_memory = True

    n_epochs = 100
    lr = 1e-4  # 0.01 for SGD
    optimizer = "Adam"

    use_gpu = True


def train(_Args=None):
    if _Args:
        Args.update(_Args)
    from jaynes.helpers import pick
    from torchvision import datasets, transforms
    import torch.nn.functional as F
    from mnist_classification.helpers import View
    from ml_logger import logger

    logger.log_params(Args=vars(Args))

    device = torch.device("cuda" if Args.use_gpu and torch.cuda.is_available() else "cpu")

    encoder = nn.Sequential(
        View(28 * 28),
        nn.Linear(28 * 28, 200),
        nn.ReLU(),
        nn.Linear(200, 40),
        nn.ReLU(),
        nn.Linear(40, 10 + 10),
    )

    decoder = nn.Sequential(
        nn.Linear(10, 40),
        nn.ReLU(),
        nn.Linear(40, 200),
        nn.ReLU(),
        nn.Linear(200, 28 * 28),
        nn.Sigmoid(),
        View(28, 28),
    )

    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    encoder.to(device)
    decoder.to(device)

    optimizer = getattr(torch.optim, Args.optimizer)([
        *encoder.parameters(), *decoder.parameters()], Args.lr)

    train_loader = DataLoader(
        datasets.MNIST('../data', train=True, download=True, transform=transforms.ToTensor()),
        batch_size=Args.batch_size, shuffle=True, **pick(vars(Args), "num_workers", "pin_memory"))
    test_loader = DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
        batch_size=Args.batch_size, shuffle=True, **pick(vars(Args), "num_workers", "pin_memory"))

    def generate_images():
        with torch.no_grad():
            for xs, _ in test_loader:
                xs = xs.to(device)
                _ = encoder(xs)
                mu, logvar = _[:, :10], _[:, 10:]
                sampled_c = reparameterize(mu, logvar)
                x_bars = decoder(sampled_c)
                break

            images = []
            for i in range(2):
                images.append(torch.cat([*xs[10 * i:10 * (i + 1)].squeeze(1)], dim=-1))
                images.append(torch.cat([*x_bars[10 * i: 10 * (i + 1)]], dim=-1))
            _ = torch.cat(images)
            _ = (_ * 255).cpu().numpy().astype('uint8')
        return _

    for epoch in range(Args.n_epochs):
        for xs, labels in tqdm(train_loader):
            xs = xs.to(device)

            _ = encoder(xs)
            mu, logvar = _[:, :10], _[:, 10:]
            sampled_c = reparameterize(mu, logvar)
            x_bars = decoder(sampled_c)
            loss = F.binary_cross_entropy(x_bars.view(-1, 784), xs.view(-1, 784), reduction='sum')
            loss += -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

            logger.store_metrics(loss=loss.item())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        _ = generate_images()
        logger.log_image(_, f"vae_{epoch:04d}.png")
        logger.log_metrics_summary(loss="min_max", key_values=dict(
            epoch=epoch, dt_epoch=logger.split()
        ))

    print('training is complete')
