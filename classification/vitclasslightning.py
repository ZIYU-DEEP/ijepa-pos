import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import RandomResizedCrop

from src.models.vision_transformer import VisionTransformer, vit_predictor, vit_small, MLP
from src.helper import init_model, load_checkpoint, init_opt
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import wandb
from lightning.pytorch.loggers.wandb import WandbLogger
from transformers import ViTForImageClassification, AdamW, ViTModel
from vit_pytorch import SimpleViT
from timm.optim.lars import Lars

#from pytorch_pretrained_vit import ViT
class IjepaCifar10(pl.LightningModule):
    def __init__(self, checkpoint_path = None, num_classes=10, learning_rate=1e-3, track_wandb=True):
        super().__init__()
        self.encoder= init_model(
            device='cuda',
            patch_size=16,
            model_name='vit_base',  # changed from vit_base
            crop_size=224,
            pred_depth=12,
            pred_emb_dim=384
        )[0]
        if checkpoint_path is not None:

            checkpoint = torch.load(checkpoint_path)

            #self.target_encoder = copy.deepcopy(encoder)
            def removeprefix(checkpointkey):
                new_state_dict = {}
                for key, value in checkpointkey.items():
                    if key.startswith('module.'):
                        new_key = key[len('module.'):]
                        new_state_dict[new_key] = value
                    else:
                        new_state_dict[key] = value
                return new_state_dict

            self.encoder.load_state_dict(removeprefix(checkpoint['encoder']))

            def remove_keys_between_indices(dictionary, start_index, end_index):
                new_dict = {k: v for i, (k, v) in enumerate(dictionary.items()) if i < start_index or i > end_index}
                return new_dict

            pred_state_dict = removeprefix(checkpoint['predictor'])
            new_pred_state_dict = remove_keys_between_indices(pred_state_dict, 76, 147)
            #self.predictor.load_state_dict(new_pred_state_dict)
        #pretrained_weights = torchvision.models.ViT_B_16_Weights.DEFAULT
        #self.encoder = torchvision.models.vit_b_16(weights=pretrained_weights)
        #self.encoder = ViT('B_16_imagenet1k', pretrained=True)
        #self.encoder = SimpleViT(   image_size = 224,
                                    # patch_size = 16,
                                    # num_classes = 10,
                                    # dim = 768,
                                    # depth = 12,
                                    # heads = 12,
                                    # mlp_dim = 3072)
        #self.avg_pooling = nn.AvgPool1d(196, stride=1)
        #self.encoder.fc = nn.Linear(in_features = self.encoder.fc.in_features, out_features=10)

        #self.batch_norm = nn.BatchNorm1d(196, affine=False, eps=1e-6)
        self.encoder.norm = torch.nn.Identity()
        #self.encoder.encoder.ln = nn.Identity()
        #self.encoder.heads = nn.Identity()
        # for param in self.encoder.parameters():
        #      param.requires_grad = False
        lin_layer = nn.Linear(in_features=768, out_features=10)
        self.head = nn.Sequential(torch.nn.BatchNorm1d(768, affine=False, eps=1e-6),lin_layer)
        for param in self.head.parameters():
            param.requires_grad = True
        #classification_head = nn.Sequential(torch.nn.BatchNorm1d(768, affine=False, eps=1e-6),lin_layer)
        #self.predictor =nn.Linear(in_features = 384, out_features=10)

        self.learning_rate = learning_rate


        self.track_wandb = track_wandb
        self.train_step_losses = []
        self.val_step_losses = []
        self.train_step_acc = []
        self.val_step_acc = []
        self.last_train_acc = 0
        self.last_train_loss = 0

    def forward(self,x):
        enc_output = self.encoder(x)
        #enc_output = self.batch_norm(enc_output)
        #enc_output = enc_output.permute(0, 2, 1)
        #pooled_output = self.avg_pooling(enc_output)
        #pooled_output = pooled_output.view(pooled_output.size(0), -1)
        pooled_output= enc_output.mean(dim=1) #average pool
        output = self.head(pooled_output)
        output = nn.functional.softmax(output, dim=1)
        return output

    def configure_optimizers(self):

        optimizer = Lars(
            self.encoder.parameters(),
            lr=0.001,
            momentum=0, #0.9,
            weight_decay=0, #0.0005,
            trust_coeff=0.001,
            eps=1e-8,
        )
        #in paper uses lr with step-wise decay, divide by factor of 10 every 15 epochs
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):

        images, labels = batch
        logits = self.forward(images)
        loss = nn.CrossEntropyLoss()(logits, labels)
        self.train_step_losses.append(loss)
        # log accuracy
        _, preds = torch.max(logits.data, dim=1)
        acc = (preds == labels).sum().item() / logits.size(dim=0)
        acc *= 100
        self.train_step_acc.append(acc)
        #visualize the input, prediction, ground truth at batch idx 0

        if batch_idx == 0:
            # Visualize the input image
            wandb.log({"input_image": [wandb.Image(images[0])]})


            pred_probabilities = logits.cpu().detach().numpy()
            plt.figure(figsize=(10, 4))
            plt.bar(range(len(pred_probabilities[0])), pred_probabilities[0])
            plt.xticks(range(len(pred_probabilities[0])))
            plt.xlabel("Class")
            plt.ylabel("Probability")
            plt.title(f"Prediction Probabilities; ground truth label: {labels[0].item()}")

            plt.savefig("prediction_probabilities.png")
            plt.close()

            wandb.log({"prediction_probabilities": [wandb.Image("prediction_probabilities.png")]})

            #wandb.log({"ground_truth": wandb.Image(labels[0])})
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

        if (self.current_epoch+1) % 15 ==0:
            lr = self.trainer.optimizers[0].param_groups[0]['lr']
            print(f'Learning rate at epoch {self.current_epoch +1}: {lr}')
        return {'train_loss': avg_loss, 'train_acc': avg_acc}

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        # enc_output = self.encoder(images.to('cuda'))
        # enc_output = enc_output.permute(0, 2, 1)
        # pooled_output = self.avg_pooling(enc_output)
        # pooled_output = pooled_output.view(pooled_output.size(0), -1)
        # pred_output = self.predictor(pooled_output)
        # pred_output_pooled = torch.mean(pred_output, dim=1)
        logits = self.forward(images)
        loss = nn.CrossEntropyLoss()(logits, labels)
        self.val_step_losses.append(loss)
        # log accuracy
        _, preds = torch.max(logits.data, dim=1)
        acc = (preds == labels).sum().item() / logits.size(dim=0)
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
    #checkpoint= '/home/christina/PycharmProjects/ijepa/logs/jepa-latest.pth.tar'
    checkpoint = None
    wandb.init(project="ijepa-classification")
    wandb_logger = WandbLogger(project="ijepa-classification")

    # transform = transforms.Compose([
    #     transforms.Resize((224, 224)),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    # ])
    transform_train = transforms.Compose([
        RandomResizedCrop(224, interpolation=3),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    transform_val = transforms.Compose([
        transforms.Resize(256, interpolation=3),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    train_dataset = CIFAR10(root="./data", train=True, download=True, transform=transform_train)
    val_dataset = CIFAR10(root="./data", train=False, download=True, transform=transform_val)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)


    model = IjepaCifar10( checkpoint_path = checkpoint)


    trainer = pl.Trainer(num_nodes=1, max_epochs=50, logger=wandb_logger)

    # Train the model
    trainer.fit(model, train_loader, val_loader)

    # Evaluate the model on val set
    # trainer.test(model, dataloaders=val_loader)
    # trainer.validate(model, dataloaders=val_loader)

