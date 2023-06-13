import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torchvision.transforms as T
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torchvision.transforms import ToTensor
from torch.optim.lr_scheduler import StepLR
import time
import torchmetrics
from tqdm import tqdm

from constants import CLASSES

EPOCHS = 50
LR = 1e-5
BATCH_SIZE = 32


class MyDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        img = self.x[index]
        return self.transform(img), self.y[index]

    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((136, 136)),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])


class TestDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        img = self.x[index]
        return self.transform(img), self.y[index]

    transform = T.Compose([
        T.ToPILImage(),
        T.ToTensor()
    ])


class MyModel(nn.Module):
    def __init__(self, num_of_classes):
        super(MyModel, self).__init__()
        # layer 1
        self.conv1 = nn.Conv2d(3, 32, kernel_size=(3, 3), stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))

        # layer 2
        self.conv2 = nn.Conv2d(32, 32, kernel_size=(3, 3), stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))

        # flatten
        self.flat = nn.Flatten()

        # layer 3
        self.fc3 = nn.Linear(36992, 128)
        self.relu3 = nn.ReLU()

        # end
        self.fc4 = nn.Linear(128, num_of_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.flat(x)

        x = self.fc3(x)
        x = self.relu3(x)

        x = self.fc4(x)
        x = self.softmax(x)

        return x


def train(model, train_loader, optimizer, criterion, device):
    model.train()
    print('Training')
    train_running_loss = 0.
    train_running_correct = 0
    counter = 0
    for i, data in tqdm(enumerate(train_loader), total=len(train_loader)):
        counter += 1
        image, labels = data
        image = image.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()

        outputs = model(image)
        loss = criterion(outputs, labels)
        train_running_loss += loss.item()

        _, predictions = torch.max(outputs.data, 1)
        train_running_correct += (predictions == labels).sum().item()

        loss.backward()
        optimizer.step()
    epoch_loss = train_running_loss / counter
    epoch_acc = 100. * (train_running_correct / len(train_loader.dataset))
    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device):
    model.eval()
    print('Validation')
    valid_running_loss = 0.0
    valid_running_correct = 0
    counter = 0
    with torch.no_grad():
        for i, data in tqdm(enumerate(val_loader), total=len(val_loader)):
            counter += 1

            image, labels = data
            image = image.to(device)
            labels = labels.to(device)

            outputs = model(image)
            loss = criterion(outputs, labels)
            valid_running_loss += loss.item()

            _, predictions = torch.max(outputs.data, 1)
            valid_running_correct += (predictions == labels).sum().item()

    epoch_loss = valid_running_loss / counter
    epoch_acc = 100. * (valid_running_correct / len(val_loader.dataset))
    return epoch_loss, epoch_acc


def train_model(x_train_set, x_val_set, y_train_set, y_val_set):
    print(torch.cuda.is_available())
    start = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = MyDataset(x_train_set, y_train_set)
    val_dataset = MyDataset(x_val_set, y_val_set)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    # model = build_model(len(CLASSES)).to(device)
    model = MyModel(len(CLASSES)).to(device)

    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = StepLR(optimizer, step_size=20, gamma=0.1)
    criterion = nn.CrossEntropyLoss()

    part1 = time.time()
    print(part1 - start)

    history = {
        "loss": [],
        "accuracy": [],
        "val_loss": [],
        "val_accuracy": []
    }

    for epoch in range(EPOCHS):
        print(f"EPOCH {epoch + 1}")
        train_epoch_loss, train_epoch_acc = train(model, train_loader, optimizer, criterion, device)
        validation_epoch_loss, validation_epoch_acc = validate(model, val_loader, criterion, device)
        history["loss"].append(train_epoch_loss)
        history["accuracy"].append(train_epoch_acc)
        history["val_loss"].append(validation_epoch_loss)
        history["val_accuracy"].append(validation_epoch_acc)

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': criterion
        }, f'outputs/model_from_epoch_{epoch + 1}.pth')

        print(
            f'Epoch {epoch + 1}: Train Loss: {train_epoch_loss:.4f} | Train Acc: {train_epoch_acc:.4f}% | Val Loss: {validation_epoch_loss:.4f} | Val Acc: {validation_epoch_acc:.2f}%'
        )
        scheduler.step()

    result = pd.DataFrame(history)
    print(result)
    print(model)

    end = time.time()
    print(end - start)
    generate_plot(history)


def build_model(num_of_classes):
    model = nn.Sequential(
        nn.Conv2d(num_of_classes, BATCH_SIZE, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Flatten(),
        nn.Linear(195840, 128),
        nn.ReLU(),
        nn.Linear(128, num_of_classes),
        nn.Softmax(dim=1)
    )
    return model


def view_training_report(model, test_ds):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)
    predictions = []
    targets = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            predictions.extend(predicted.cpu().numpy())
            targets.extend(labels.cpu().numpy())

    y_pred = np.array(predictions)
    y_true = np.array(targets)

    accuracy = torchmetrics.functional.accuracy(
        torch.tensor(y_pred),
        torch.tensor(y_true),
        task="multiclass",
        num_classes=len(CLASSES),
        average="none"
    )
    precision = torchmetrics.functional.precision(
        torch.tensor(y_pred),
        torch.tensor(y_true),
        num_classes=len(CLASSES),
        task="multiclass",
        average="none"
    )
    recall = torchmetrics.functional.recall(
        torch.tensor(y_pred),
        torch.tensor(y_true),
        num_classes=len(CLASSES),
        task="multiclass",
        average="none"
    )
    f1_score = torchmetrics.functional.f1_score(
        torch.tensor(y_pred),
        torch.tensor(y_true),
        num_classes=len(CLASSES),
        task="multiclass",
        average="none"
    )
    confusion_mat = torchmetrics.functional.confusion_matrix(
        torch.tensor(y_pred),
        torch.tensor(y_true),
        num_classes=len(CLASSES),
        task="multiclass"
    )
    print("CLASSIFICATION REPORT: ")
    print(f"ACCURACY: {accuracy}")
    print(f"PRECISION: {precision}")
    print(f"RECALL: {recall}")
    print(f"F1 SCORE: {f1_score}")
    print(f"CONFUSION MATRIX: {confusion_mat.data}")


def generate_plot(history):
    acc = history["accuracy"]
    val_acc = history["val_accuracy"]
    loss = history["loss"]
    val_loss = history["val_loss"]
    epochs = range(1, len(acc) + 1)

    plt.plot(epochs, acc, label="Training", color="green")
    plt.plot(epochs, val_acc, label="Validation", color="red")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.figure()

    plt.plot(epochs, loss, label='Training', color='green')
    plt.plot(epochs, val_loss, label='Validation', color='red')
    plt.title('Loss during Training and Validation')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()


def test_model(model_path, x_test_dataset, y_test_dataset):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MyModel(len(CLASSES)).to(device)
    loaded = torch.load(model_path)
    model.load_state_dict(loaded["model_state_dict"])
    # model = torch.load(model_path)
    dataset = MyDataset(x_test_dataset, y_test_dataset)
    view_training_report(model, dataset)


def test(model, loader, device):
    model.eval()
    print("Testing")
    test_running_correct = 0
    counter = 0
    with torch.no_grad():
        for i, data in tqdm(enumerate(loader), total=len(loader)):
            counter += 1

            image, labels = data
            image = image.to(device)
            labels = labels.to(device)

            outputs = model(image)

            _, predictions = torch.max(outputs.data, 1)
            test_running_correct += (predictions == labels).sum().item()

    accuracy = 100. * (test_running_correct / len(loader.dataset))
    return accuracy
