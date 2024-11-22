import argparse
import json
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.cuda.amp as amp
from torch.optim.lr_scheduler import StepLR
from torchvision import transforms, models
from medmnist import PathMNIST
import determined as det
from torch.utils.data import DataLoader, Subset
import numpy as np
import logging
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import pairwise_distances_argmin_min, accuracy_score, roc_auc_score

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

SAVE_DIRECTORY = "pytorch_model"
os.makedirs(SAVE_DIRECTORY, exist_ok=True)

class ResNet50_28(nn.Module):
    def __init__(self, num_classes):
        super(ResNet50_28, self).__init__()
        self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.maxpool = nn.Identity()
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

def train(args, model, device, train_loader, optimizer, epoch_idx, core_context, steps_completed, scaler):
    model.train()
    all_targets, all_predictions = [], []
    epoch_loss = 0
    num_samples = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        if data.size(0) == 0:
            continue
        data, target = data.to(device), target.to(device).squeeze()
        optimizer.zero_grad()

        with amp.autocast():
            output = model(data)
            loss = nn.CrossEntropyLoss(label_smoothing=0.1)(output, target)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        steps_completed += 1

        epoch_loss += loss.item() * data.size(0)
        num_samples += data.size(0)
        probabilities = torch.softmax(output, dim=1).detach().cpu().numpy()
        all_targets.extend(target.cpu().numpy())
        all_predictions.extend(probabilities)

    epoch_loss /= num_samples
    train_accuracy = accuracy_score(all_targets, np.argmax(all_predictions, axis=1))

    logger.info(
        f"Train Epoch {epoch_idx} Complete: Loss: {epoch_loss:.4f}, Accuracy: {train_accuracy:.4f}"
    )
    core_context.train.report_training_metrics(
        steps_completed=steps_completed,
        metrics={"train_loss": epoch_loss, "train_accuracy": train_accuracy},
    )
    return steps_completed

def test(args, model, device, test_loader, core_context, steps_completed, epochs_completed):
    model.eval()
    test_loss = 0
    all_targets, all_predictions = [], []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device).squeeze()
            output = model(data)
            test_loss += nn.CrossEntropyLoss(reduction="sum")(output, target).item()
            probabilities = torch.softmax(output, dim=1).detach().cpu().numpy()
            all_targets.extend(target.cpu().numpy())
            all_predictions.extend(probabilities)

    test_loss /= len(test_loader.dataset)
    accuracy = accuracy_score(all_targets, np.argmax(all_predictions, axis=1))
    auc = roc_auc_score(all_targets, all_predictions, multi_class='ovr')

    logger.info(f"Test set: Average loss: {test_loss:.4f}, Accuracy: {accuracy:.4f}, AUC: {auc:.4f}")
    core_context.train.report_validation_metrics(
        steps_completed=steps_completed,
        metrics={"test_loss": test_loss, "test_accuracy": accuracy, "test_auc": auc, "epochs": epochs_completed},
    )
    return test_loss

def mixed_sampling_strategy(model, unlabeled_loader, device, num_samples, mc_dropout_passes=5):
    model.train()
    uncertainties, embeddings, indices = [], [], []

    with torch.no_grad():
        for batch_idx, (images, _) in enumerate(unlabeled_loader):
            images = images.to(device)
            mc_outputs = torch.stack([
                torch.softmax(model(images), dim=1) for _ in range(mc_dropout_passes)
            ], dim=0)

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

def save_model_to_huggingface_format(model, num_classes, input_size=(3, 28, 28), save_directory=SAVE_DIRECTORY):
    """Save the model and configuration in Hugging Face-compatible format."""
    try:
        os.makedirs(save_directory, exist_ok=True)

        model_save_path = os.path.join(save_directory, "pytorch_model.bin")
        torch.save(model.state_dict(), model_save_path)
        config = {
            "model_type": "resnet50",
            "num_classes": num_classes,
            "input_size": input_size,
            "architecture": "ResNet50",
        }
        config_save_path = os.path.join(save_directory, "config.json")
        with open(config_save_path, "w") as f:
            json.dump(config, f, indent=4)

        logger.info(f"Model and configuration saved in Hugging Face format at {save_directory}")
    except Exception as e:
        logger.error(f"Failed to save Hugging Face-compatible model: {e}")

def main(core_context):
    global SAVE_DIRECTORY
    parser = argparse.ArgumentParser(description="Active Learning on MedMNIST")
    parser.add_argument("--batch-size", type=int, default=64, metavar="N", help="Training batch size")
    parser.add_argument("--test-batch-size", type=int, default=1000, metavar="N", help="Testing batch size")
    parser.add_argument("--epochs", type=int, default=5, metavar="N", help="Number of epochs to train")
    parser.add_argument("--lr", type=float, default=0.01, metavar="LR", help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.7, metavar="M", help="Learning rate step gamma")
    parser.add_argument("--log-interval", type=int, default=100, metavar="N", help="Log interval")
    parser.add_argument("--dry-run", action="store_true", default=False, help="Run a single pass for debugging")
    args = parser.parse_args()

    logger.info("Initializing the experiment...")
    info = det.get_cluster_info()
    assert info is not None, "This script must be run on-cluster with Determined AI"
    logger.info(f"Running on Determined AI with cluster info: {info}")

    latest_checkpoint = info.latest_checkpoint

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    os.makedirs(SAVE_DIRECTORY, exist_ok=True)

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
    
    logger.info("Loading datasets...")
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

    train_loader = DataLoader(labeled_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(val_dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=4)

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
    scaler = amp.GradScaler()

    if latest_checkpoint is None:
        epochs_completed = 0
        best_val_loss = float('inf')
    else:
        with core_context.checkpoint.restore_path(latest_checkpoint) as path:
            checkpoint = torch.load(path / "checkpoint.pt")
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            epochs_completed = checkpoint["epochs_completed"]
            best_val_loss = checkpoint.get("best_val_loss", float('inf'))
            logger.info(f"Resumed from checkpoint. Epochs completed: {epochs_completed}, Best val loss: {best_val_loss:.4f}")

    steps_completed = 0

    for iteration in range(active_learning_iterations):
        logger.info(f"Active Learning Iteration {iteration + 1}/{active_learning_iterations}")

        for epoch_idx in range(args.epochs):
            steps_completed = train(args, model, device, train_loader, optimizer, epoch_idx, core_context, steps_completed, scaler)
            val_loss = test(args, model, device, test_loader, core_context, steps_completed, iteration + 1)
            if val_loss is None:
                val_loss = float('inf')
            scheduler.step(val_loss)

        logger.info("Performing active learning sampling...")
        selected_indices = mixed_sampling_strategy(model, unlabeled_loader, device, num_active_learning_samples)

        labeled_indices.extend(selected_indices)
        unlabeled_indices = list(set(unlabeled_indices) - set(selected_indices))

        labeled_dataset = Subset(full_train_dataset, labeled_indices)
        unlabeled_dataset = Subset(full_train_dataset, unlabeled_indices)

        train_loader = DataLoader(labeled_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
        unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

        logger.info(f"Iteration {iteration + 1}: Labeled samples: {len(labeled_indices)}, Unlabeled samples: {len(unlabeled_indices)}")

        checkpoint_metadata_dict = {"epochs_completed": iteration + 1, "steps_completed": steps_completed}
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            with core_context.checkpoint.store_path(checkpoint_metadata_dict) as (path, storage_id):
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "epochs_completed": iteration + 1,
                        "best_val_loss": best_val_loss,
                    },
                    os.path.join(path, "checkpoint.pt"),
                )
                hf_save_directory = os.path.join(path, "huggingface")
                save_model_to_huggingface_format(model, num_classes=9, save_directory=hf_save_directory)
                logger.info(f"Hugging Face-compatible model saved in checkpoint directory: {hf_save_directory}")

        if core_context.preempt.should_preempt():
            logger.info("Preempting job...")
            return

if __name__ == "__main__":
    with det.core.init() as core_context:
        main(core_context)