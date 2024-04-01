"""Here are 4 easy steps to use Fabric in your PyTorch code.

1. Create the Lightning Fabric object at the beginning of your script.

2. Remove all ``.to`` and ``.cuda`` calls since Fabric will take care of it.

3. Apply ``setup`` over each model and optimizers pair, ``setup_dataloaders`` on all your dataloaders,
and replace ``loss.backward()`` with ``self.backward(loss)``.

4. Run the script from the terminal using ``fabric run path/to/train.py``

Accelerate your training loop by setting the ``--accelerator``, ``--strategy``, ``--devices`` options directly from
the command line. See ``fabric run --help`` or learn more from the documentation:
https://lightning.ai/docs/fabric.

"""

import argparse
from os import path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T
from lightning.fabric import Fabric, seed_everything
from torch.optim.lr_scheduler import StepLR
from torchmetrics.classification import Accuracy
from torchvision.datasets import MNIST


class ECG_PPG_Dataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.sample_files = list(self.data_dir.glob("feats_*.npy"))
        self.target_files = list(self.data_dir.glob("targets_*.npy"))

        self.sample_maps = []
        self.target_maps = []
        self.sample_indices = []

        for i, (sample_file, target_file) in enumerate(
            zip(self.sample_files, self.target_files)
        ):
            sample_map = np.load(sample_file, mmap_mode="r")
            target_map = np.load(target_file, mmap_mode="r")
            logging.info(f"load {sample_file} and {target_file}")

            self.sample_maps.append(sample_map)
            self.target_maps.append(target_map)

            for j in range(len(sample_map)):
                self.sample_indices.append((i, j))

    def __len__(self):
        return len(self.sample_indices)

    def __getitem__(self, idx):
        file_index, sample_index = self.sample_indices[idx]
        sample = self.sample_maps[file_index][sample_index].copy()
        sample = sample.reshape(1, -1)
        target = self.target_maps[file_index][sample_index].copy()

        if self.transform:
            sample = self.transform.augment(sample)

        return torch.from_numpy(sample).float(), torch.from_numpy(target).long()


class Net(nn.Module):
    def __init__(self, input_size=2216, output_size=104) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, 3, 1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Flatten(),
            nn.LazyLinear(1024),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, output_size),
        )

    def forward(self, x):
        return self.net(x)


def run(hparams):
    # Create the Lightning Fabric object. The parameters like accelerator, strategy, devices etc. will be proided
    # by the command line. See all options: `fabric run --help`
    fabric = Fabric()

    seed_everything(hparams.seed)  # instead of torch.manual_seed(...)

    # Let rank 0 download the data first
    with fabric.rank_zero_first(
        local=False
    ):  # set `local=True` if your filesystem is not shared between machines
        dataset = ECG_PPG_Dataset(
            data_dir="/work/app/wanghongyang/dataset/ecg_ppg_feat_map", transform=None
        )
        train_set, val_set = torch.utils.data.random_split(dataset, [0.8, 0.2])

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=hparams.batch_size,
    )
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=hparams.batch_size)

    # don't forget to call `setup_dataloaders` to prepare for dataloaders for distributed training.
    train_loader, val_loader = fabric.setup_dataloaders(train_loader, val_loader)

    model = Net()  # remove call to .to(device)
    optimizer = optim.AdamW(model.parameters(), lr=hparams.lr)

    # don't forget to call `setup` to prepare for model / optimizer for distributed training.
    # the model is moved automatically to the right device.
    model, optimizer = fabric.setup(model, optimizer)

    scheduler = StepLR(optimizer, step_size=1, gamma=hparams.gamma)

    # use torchmetrics instead of manually computing the accuracy
    test_acc = Accuracy(task="multiclass", num_classes=10).to(fabric.device)

    # EPOCH LOOP
    for epoch in range(1, hparams.epochs + 1):
        # TRAINING LOOP
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            # NOTE: no need to call `.to(device)` on the data, target
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            fabric.backward(loss)  # instead of loss.backward()

            optimizer.step()
            if (batch_idx == 0) or ((batch_idx + 1) % hparams.log_interval == 0):
                print(
                    f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}"
                    f" ({100.0 * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}"
                )
                if hparams.dry_run:
                    break

        scheduler.step()

        # TESTING LOOP
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for data, target in val_loader:
                # NOTE: no need to call `.to(device)` on the data, target
                output = model(data)
                test_loss += F.nll_loss(output, target, reduction="sum").item()

                # WITHOUT TorchMetrics
                # pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                # correct += pred.eq(target.view_as(pred)).sum().item()

                # WITH TorchMetrics
                test_acc(output, target)

                if hparams.dry_run:
                    break

        # all_gather is used to aggregated the value across processes
        test_loss = fabric.all_gather(test_loss).sum() / len(val_loader.dataset)

        print(
            f"\nTest set: Average loss: {test_loss:.4f}, Accuracy: ({100 * test_acc.compute():.0f}%)\n"
        )
        test_acc.reset()

        if hparams.dry_run:
            break

    # When using distributed training, use `fabric.save`
    # to ensure the current process is allowed to save a checkpoint
    if hparams.save_model:
        fabric.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == "__main__":
    # Arguments can be passed in through the CLI as normal and will be parsed here
    # Example:
    # fabric run image_classifier.py accelerator=cuda --epochs=3
    parser = argparse.ArgumentParser(description="Fabric MNIST Example")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=14,
        metavar="N",
        help="number of epochs to train (default: 14)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1.0,
        metavar="LR",
        help="learning rate (default: 1.0)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.7,
        metavar="M",
        help="Learning rate step gamma (default: 0.7)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="quickly check a single pass",
    )
    parser.add_argument(
        "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    parser.add_argument(
        "--save-model",
        action="store_true",
        default=False,
        help="For Saving the current Model",
    )
    hparams = parser.parse_args()

    run(hparams)
