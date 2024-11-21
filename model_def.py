import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torchvision import transforms, models
from medmnist import PathMNIST
import determined as det
from torch.utils.data import DataLoader, Subset
import numpy as np
import logging
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import pairwise_distances_argmin_min

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class ResNet50_28(nn.Module):
    def __init__(self, num_classes):
        super(ResNet50_28, self).__init__()
        self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.maxpool = nn.Identity()
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

def train(args, model, device, train_loader, optimizer, epoch_idx, core_context, steps_completed):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if data.size(0) == 0:  # Skip empty batches
            continue
        data, target = data.to(device), target.to(device).squeeze()
        optimizer.zero_grad()
        output = model(data)
        if target.size(0) != output.size(0):
            raise ValueError(f"Mismatch: input batch_size ({output.size(0)}) != target batch_size ({target.size(0)})")
        loss = nn.CrossEntropyLoss()(output, target)
        loss.backward()
        optimizer.step()

        steps_completed += 1

        if (batch_idx + 1) % args.log_interval == 0:
            logger.info(
                f"Train Epoch: {epoch_idx} [{batch_idx * len(data)}/{len(train_loader.dataset)}"
                f" ({100.0 * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}"
            )
            core_context.train.report_training_metrics(
                steps_completed=steps_completed,
                metrics={"train_loss": loss.item()},
            )
    return steps_completed



def test(args, model, device, test_loader, core_context, steps_completed, epochs_completed):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device).squeeze()
            output = model(data)
            test_loss += nn.CrossEntropyLoss(reduction="sum")(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    logger.info(
        f"Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)}"
        f" ({100.0 * correct / len(test_loader.dataset):.0f}%)"
    )
    core_context.train.report_validation_metrics(
        steps_completed=steps_completed,
        metrics={"test_loss": test_loss, "epochs": epochs_completed},
    )
    return test_loss

def mixed_sampling_strategy(model, unlabeled_loader, device, num_samples, mc_dropout_passes=5):
    model.train()
    uncertainties, embeddings, indices = [], [], []

    with torch.no_grad():
        for batch_idx, (images, _) in enumerate(unlabeled_loader):
            images = images.to(device)
            mc_outputs = torch.zeros(mc_dropout_passes, images.size(0), 9).to(device)
            for i in range(mc_dropout_passes):
                mc_outputs[i] = torch.softmax(model(images), dim=1)

            mean_output = mc_outputs.mean(dim=0)
            entropy = -torch.sum(mean_output * torch.log(mean_output + 1e-6), dim=1)
            uncertainties.extend(entropy.cpu().numpy())
            embeddings.extend(mean_output.cpu().numpy())
            indices.extend(batch_idx * len(images) + np.arange(len(images)))

    uncertainties = np.array(uncertainties)
    embeddings = np.array(embeddings)
    indices = np.array(indices)

    high_uncertainty_indices = np.argsort(-uncertainties)[: num_samples // 2]

    kmeans = MiniBatchKMeans(n_clusters=num_samples // 2, random_state=0, batch_size=100).fit(embeddings)
    diverse_indices, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, embeddings)

    selected_indices = np.unique(np.concatenate([high_uncertainty_indices, diverse_indices]))
    return indices[selected_indices]

def main(core_context):
    parser = argparse.ArgumentParser(description="Active Learning on MedMNIST")
    parser.add_argument("--batch-size", type=int, default=64, metavar="N", help="Training batch size")
    parser.add_argument("--test-batch-size", type=int, default=1000, metavar="N", help="Testing batch size")
    parser.add_argument("--epochs", type=int, default=5, metavar="N", help="Number of epochs to train")
    parser.add_argument("--lr", type=float, default=0.01, metavar="LR", help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.7, metavar="M", help="Learning rate step gamma")
    parser.add_argument("--log-interval", type=int, default=100, metavar="N", help="Log interval")
    parser.add_argument("--dry-run", action="store_true", default=False, help="Run a single pass for debugging")
    args = parser.parse_args()

    info = det.get_cluster_info()
    assert info is not None, "This script must be run on-cluster with Determined AI"
    latest_checkpoint = info.latest_checkpoint

    torch.manual_seed(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.RandomResizedCrop(28, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomApply([transforms.GaussianBlur(3)], p=0.5),
        transforms.RandAugment(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5])
    ])
    full_train_dataset = PathMNIST(split="train", transform=transform, download=True)
    val_dataset = PathMNIST(split="val", transform=transform, download=True)

    hparams = info.trial.hparams
    initial_labeled_size = hparams.get("initial_labeled_size", 4000)
    num_active_learning_samples = hparams.get("samples_to_label", 500)
    active_learning_iterations = hparams.get("active_learning_iterations", 20)

    indices = np.arange(len(full_train_dataset))
    np.random.shuffle(indices)
    labeled_indices = indices[:initial_labeled_size].tolist()
    unlabeled_indices = indices[initial_labeled_size:].tolist()

    labeled_dataset = Subset(full_train_dataset, labeled_indices)
    unlabeled_dataset = Subset(full_train_dataset, unlabeled_indices)

    train_loader = DataLoader(labeled_dataset, batch_size=args.batch_size, shuffle=True)
    unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(val_dataset, batch_size=args.test_batch_size, shuffle=False)

    model = ResNet50_28(num_classes=9).to(device)
    optimizer = optim.SGD(
        model.parameters(),
        lr=hparams.get("learning_rate", 0.01),
        momentum=0.9,
        weight_decay=hparams.get("weight_decay", 0.0005)
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=5, verbose=True
    )

    if latest_checkpoint is None:
        epochs_completed = 0
    else:
        with core_context.checkpoint.restore_path(latest_checkpoint) as path:
            checkpoint = torch.load(path / "checkpoint.pt")
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            epochs_completed = checkpoint["epochs_completed"]

    steps_completed = 0 
    
    for iteration in range(active_learning_iterations):
        logger.info(f"Active Learning Iteration {iteration + 1}")

        for epoch_idx in range(args.epochs):
            steps_completed = train(args, model, device, train_loader, optimizer, epoch_idx, core_context, steps_completed)
            
            val_loss = test(args, model, device, test_loader, core_context, steps_completed, iteration + 1)
            if val_loss is None:  # Default value if None
                val_loss = float("inf")
            scheduler.step(val_loss)

        selected_indices = mixed_sampling_strategy(model, unlabeled_loader, device, num_active_learning_samples)

        labeled_indices.extend(selected_indices)
        unlabeled_indices = list(set(unlabeled_indices) - set(selected_indices))

        labeled_dataset = Subset(full_train_dataset, labeled_indices)
        unlabeled_dataset = Subset(full_train_dataset, unlabeled_indices)

        train_loader = DataLoader(labeled_dataset, batch_size=args.batch_size, shuffle=True)
        unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=args.batch_size, shuffle=False)

        logger.info(f"Iteration {iteration + 1}: Labeled samples: {len(labeled_indices)}, Unlabeled samples: {len(unlabeled_indices)}")

        checkpoint_metadata_dict = {"epochs_completed": iteration + 1, "steps_completed": steps_completed}
        with core_context.checkpoint.store_path(checkpoint_metadata_dict) as (path, storage_id):
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epochs_completed": iteration + 1,
                },
                path / "checkpoint.pt",
            )

        if core_context.preempt.should_preempt():
            return

if __name__ == "__main__":
    with det.core.init() as core_context:
        main(core_context)