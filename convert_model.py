from huggingface_hub import Repository
import torch
import os
import torch.nn as nn
from torchvision import transforms, models


model_path = "determined_checkpoints/medmnist_checkpoints/5088f8da-1f5a-419e-974c-f9551e57bdcb/checkpoint.pt"
save_directory = "./Medmnist_active_learning"

class ResNet50_28(nn.Module):
    def __init__(self, num_classes):
        super(ResNet50_28, self).__init__()
        self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.maxpool = nn.Identity()
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

model = ResNet50_28(num_classes=9)
checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint["model_state_dict"])

if not os.path.exists(save_directory):
    os.makedirs(save_directory)
model_save_path = os.path.join(save_directory, "pytorch_model.bin")
torch.save(model.state_dict(), model_save_path)

readme_path = os.path.join(save_directory, "README.md")
with open(readme_path, "w") as f:
    f.write("# MedMNIST Active Learning Model\n\nCustom ResNet50 model trained using active learning strategies.")

repo_url = "https://huggingface.co/alcheung0213/Medmnist_active_learning"
repo = Repository(local_dir=save_directory, clone_from=repo_url)

repo.push_to_hub(commit_message="Initial upload of MedMNIST Active Learning model")
