import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
from argparse import ArgumentParser
from pytorch_lightning.core.lightning import LightningModule
from modules.spatial_transformer import ST
from modules.local_contrast_normalizer import LCN
from modules.view import View
from data_loader import get_train_dataset, get_train_and_val_dataset


class StnCnn(LightningModule):
    def __init__(self, hparams):
        super(StnCnn, self).__init__()

        self.hparams = hparams
        self.train_set = None
        self.val_set = None

        self.feature = nn.Sequential(
            ST((3, 48), (250, 250, 250)),
            nn.Conv2d(3, 200, kernel_size=7, padding=2),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
            LCN((1, 200, 7, 7)),
            ST((200, 23), (150, 200, 300)),
            nn.Conv2d(200, 250, kernel_size=4, padding=2),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
            LCN((1, 250, 7, 7)),
            ST((250, 12), (150, 200, 300)),
            nn.Conv2d(250, 350, kernel_size=4, padding=2),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
            LCN((1, 350, 7, 7))
        )

        self.classifier = nn.Sequential(
            View(350 * 6 * 6),
            nn.Linear(350 * 6 * 6, 400),
            nn.ReLU(True),
            nn.Linear(400, 19)
        )

    def forward(self, x):
        # transform the input
        x = self.feature(x)
        x = self.classifier(x)
        return x

    # Data
    def prepare_data(self):
        if self.hparams.validate:
            train_set, val_set = get_train_and_val_dataset(self.hparams.train_dir, self.hparams.train_ratio)
            self.train_set = train_set
            self.val_set = val_set
        else:
            self.val_dataloader = None
            self.validation_step = None
            self.validation_epoch_end = None
            train_set = get_train_dataset(self.hparams.train_dir)
            self.train_set = train_set

    def train_dataloader(self):
        train_loader = torch.utils.data.DataLoader(self.train_set, batch_size=self.hparams.batch_size, shuffle=True)
        return train_loader

    def val_dataloader(self):
        val_loader = torch.utils.data.DataLoader(self.val_set, batch_size=self.hparams.batch_size, shuffle=False)
        return val_loader

    # Training
    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = F.cross_entropy(outputs, labels)
        # self.logger.summary.scalar('loss', loss)
        logs = {'loss': loss}
        return {'loss': loss, 'log': logs}

    # Validation
    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = F.cross_entropy(outputs, labels)
        # calc accuracy
        correct = 0
        total = 0
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        return {'val_loss': loss, 'val_correct': correct, 'val_total': total}

    def validation_epoch_end(self, val_results):
        avg_loss = torch.stack([x['val_loss'] for x in val_results]).mean()
        total_correct = torch.Tensor([x['val_correct'] for x in val_results]).sum()
        total = torch.Tensor([x['val_total'] for x in val_results]).sum()
        avg_accuracy = total_correct / total
        tensorboard_logs = {'val_loss': avg_loss, 'val_accuracy': avg_accuracy}
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    # Test
    def test_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = F.cross_entropy(outputs, labels)
        return {'val_loss': loss}

    def test_epoch_end(self, val_losses):
        avg_loss = torch.stack([x['val_loss'] for x in val_losses]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--train_dir", type=str, required=True)
        parser.add_argument("--validate", action="store_true")
        parser.add_argument("--train_ratio", type=float, default=0.8)
        parser.add_argument("--batch_size", type=int, default=50)
        parser.add_argument("--learning_rate", type=float, default=0.01)
        return parser
