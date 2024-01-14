import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from src.models.vision_transformer import VisionTransformer, vit_predictor, vit_small, MLP
from src.helper import init_model, load_checkpoint, init_opt
import matplotlib.pyplot as plt
checkpoint = torch.load('/home/christina/PycharmProjects/ijepa/logs/jepa-latest.pth.tar')

encoder, predictor = init_model(
    device='cuda',
    patch_size=16,
    model_name='vit_small', #changed from vit_base
    crop_size=224,
    pred_depth=6,
    pred_emb_dim=384
)
target_encoder = copy.deepcopy(encoder)

#reformat keys in state_dict
def removeprefix(checkpointkey):
    new_state_dict = {}
    for key, value in checkpointkey.items():
        if key.startswith('module.'):
            new_key = key[len('module.'):]
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    return new_state_dict

encoder.load_state_dict(removeprefix(checkpoint['encoder']))

def remove_keys_between_indices(dictionary, start_index, end_index):
    new_dict = {k: v for i, (k, v) in enumerate(dictionary.items()) if i < start_index or i > end_index}
    return new_dict
pred_state_dict = removeprefix(checkpoint['predictor'])
new_pred_state_dict = remove_keys_between_indices(pred_state_dict, 76, 147)
predictor.load_state_dict(new_pred_state_dict)

target_encoder.load_state_dict(removeprefix(checkpoint['target_encoder']))

#embed_dim = 384
classification_head = nn.Linear(in_features=predictor.embed_dim, out_features = 10)
predictor = classification_head
#target_encoder = nn.Identity()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

classification_dataset = CIFAR10(root='data', train=True, transform=transform, download=True)
classification_dataloader = DataLoader(classification_dataset, batch_size=32, shuffle=True)


optimizer = optim.Adam(predictor.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

num_epochs = 1
loss_values = []
for epoch in range(num_epochs):
    encoder.train()
    predictor.train()
    for batch_idx, batch in enumerate(classification_dataloader):
        inputs, targets = batch

        optimizer.zero_grad()

        # Forward pass
        enc_output = encoder(inputs.to('cuda'))
        pred_output = predictor(enc_output.to('cpu'))
        pred_output_pooled = torch.mean(pred_output, dim=1)
        #print(logits.shape)
        loss = criterion(pred_output_pooled, targets)

        # Backpropagation and optimization
        loss.backward()
        optimizer.step()

        if (batch_idx + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(classification_dataloader)}], Loss: {loss.item()}')
    loss_values.append(loss.item())

epochs = range(1, len(loss_values) + 1)


plt.plot(epochs, loss_values, 'b', label='Training Loss')
plt.title('Training Loss Curve')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('finetuned model loss')
plt.show()

torch.save(encoder.state_dict(), 'fine_tuned_model.pth')


#begin eval
testset = CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=2)

encoder.eval()
predictor.eval()

correct = 0
total = 0
loss_sum = 0

with torch.no_grad():
    for batch_idx, batch in enumerate(classification_dataloader):
        inputs, targets = batch
        enc_output = encoder(inputs.to('cuda'))
        pred_output = predictor(enc_output.to('cpu'))
        pred_output_pooled = torch.mean(pred_output, dim=1)
        loss = criterion(pred_output_pooled, targets)

        _, predicted = torch.max(pred_output_pooled, 1)

        total += targets.size(0)
        correct += (predicted == targets).sum().item()
        loss_sum += loss.item()

accuracy = 100 * correct/total
avg_loss = loss_sum/len(testloader)

print(f"accuracy of fine-tuned model on test set: {accuracy:.2f}")
print(f"average loss: {avg_loss:.4f}")
