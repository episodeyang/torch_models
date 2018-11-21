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

    n_epochs = 200
    lr = 1e-4  # 0.01 for SGD
    optimizer = "Adam"


def train(_Args=None):
    if _Args:
        Args.update(_Args)
    from jaynes.helpers import pick
    from torchvision import datasets, transforms
    import torch.nn.functional as F
    from mnist_classification.helpers import View
    from ml_logger import logger

    logger.log_params(Args=vars(Args))

    mlp = nn.Sequential(
        View(28 * 28),
        nn.Linear(28 * 28, 200),
        nn.ReLU(),
        nn.Linear(200, 40),
        nn.ReLU(),
        nn.Linear(40, 10),
        nn.Softmax(dim=-1)
    )

    optimizer = getattr(torch.optim, Args.optimizer)(mlp.parameters(), Args.lr)

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = DataLoader(
        datasets.MNIST('../data', train=True, download=True, transform=transforms.ToTensor()),
        batch_size=Args.batch_size, shuffle=True, **pick(vars(Args), "num_workers", "pin_memory"))
    test_loader = DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
        batch_size=Args.batch_size, shuffle=True, **pick(vars(Args), "num_workers", "pin_memory"))

    def test():
        with torch.no_grad():
            for xs, lables in test_loader:
                logits = mlp(xs)
                _, ys = logits.max(dim=-1)
                logger.store_metrics(accuracy=(ys == lables).float().mean().item())

    for epoch in range(Args.n_epochs):

        for xs, labels in tqdm(train_loader):
            ys = mlp(xs)
            loss = F.nll_loss(ys, labels)
            logger.store_metrics(loss=loss.item())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        test()
        logger.log_metrics_summary(accuracy='mean', loss="min_max", key_values=dict(
            epoch=epoch, dt_epoch=logger.split()
        ))

    print('training is complete')


if __name__ == "__main__":
    import jaynes
    from mnist_classification.thunk import thunk
    from ml_logger import logger

    jaynes.config(runner=dict(n_cpu=40))
    jaynes.run(thunk(train,
                     log_dir="http://54.71.92.65:8081",
                     log_prefix=f"debug/mnist_classification/{logger.stem(__file__)}/{logger.now('%f')}"),
               )
    jaynes.listen()
