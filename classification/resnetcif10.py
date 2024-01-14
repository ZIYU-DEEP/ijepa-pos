import torch
import torch.nn as nn
import torchvision
import pytorch_lightning as pl
import wandb
from lightning.pytorch.loggers import WandbLogger
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets


class ResNetCIFAR10(pl.LightningModule):
    def __init__(self, num_classes=10, learning_rate=1e-3, track_wandb=True):
        super().__init__()
        self.model = torchvision.models.resnet50(pretrained=False)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)
        self.learning_rate = learning_rate

        self.track_wandb = track_wandb
        self.train_step_losses = []
        self.val_step_losses = []
        self.train_step_acc = []
        self.val_step_acc = []
        self.last_train_acc = 0
        self.last_train_loss = 0

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def training_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        self.train_step_losses.append(loss)
        # log accuracy
        _, preds = torch.max(outputs.data, dim=1)
        acc = (preds == labels).sum().item() / outputs.size(dim=0)
        acc *= 100
        self.train_step_acc.append(acc)

        return {'loss': loss}

    def on_train_epoch_end(self):
        all_preds = self.train_step_losses
        avg_loss = sum(all_preds) / len(all_preds)

        all_acc = self.train_step_acc
        avg_acc = sum(all_acc) / len(all_acc)
        avg_acc = round(avg_acc, 2)

        self.last_train_loss = avg_loss
        self.last_train_acc = avg_acc

        self.train_step_acc.clear()
        self.train_step_losses.clear()

        return {'train_loss': avg_loss, 'train_acc': avg_acc}

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        self.val_step_losses.append(loss)
        # log accuracy
        _, preds = torch.max(outputs.data, dim=1)
        acc = (preds == labels).sum().item() / outputs.size(dim=0)
        acc *= 100
        self.val_step_acc.append(acc)

        return {'val_loss': loss}

    def on_validation_epoch_end(self):
        all_preds = self.val_step_losses
        all_acc = self.val_step_acc

        avg_loss = sum(all_preds) / len(all_preds)
        avg_acc = sum(all_acc) / len(all_acc)
        # avg_acc = round(avg_acc, 2)

        if self.track_wandb:
            wandb.log({"training_loss": self.last_train_loss,
                       "training_acc": self.last_train_acc,
                       "validation_loss": avg_loss,
                       "validation_acc": avg_acc})

        self.val_step_losses.clear()
        self.val_step_acc.clear()

        return {'val_loss': avg_loss, 'val_acc': avg_acc}

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = nn.CrossEntropyLoss()(logits, y)
        self.log("test_loss", loss)


if __name__ == "__main__":
    # pl.seed_everything(42)


    wandb.init(project="cifar10-classification")
    wandb_logger = WandbLogger(project="cifar10-classification")

    transform = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.5,), (0.5,))]
    )
    train_dataset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    val_dataset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)


    model = ResNetCIFAR10()


    trainer = pl.Trainer(num_nodes=1, max_epochs=50, logger=wandb_logger)

    # Train the model
    trainer.fit(model, train_loader, val_loader)

    # Evaluate the model on val set
    # trainer.test(model, dataloaders=val_loader)
    # trainer.validate(model, dataloaders=val_loader)
