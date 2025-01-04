import random
from functools import partial

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import datasets, transforms
def get_modified_dataset(dataset):
    """
    Return a modified dataset cls that can insert MNIST like images into larger
    frames with an option for random shifts, scrambling the images in a
    consistent way (using the same shuffling for all images) and adding random
    Gaussian noise (to the base data, noise is always the same for a given
    image).
    """

    class ModifiedDataset(dataset):
        def __init__(
            self,
            root,
            img_size=56,
            random_shift=False,
            scramble_image=False,
            noise=0.0,
            *args,
            **kwargs
        ):
            super().__init__(root, *args, **kwargs)
            assert img_size >= 28
            self.img_size = img_size
            self.scramble_image = scramble_image
            assert noise >= 0.0
            self.noise = noise

            if random_shift:
                rng = random.Random(433)
                self.r_idxs = [
                    rng.randrange(img_size - 28 + 1) for _ in range(len(self))
                ]
                self.c_idxs = [
                    rng.randrange(img_size - 28 + 1) for _ in range(len(self))
                ]
            else:
                self.r_idxs = [(img_size - 28) // 2] * len(self)
                self.c_idxs = self.r_idxs
            self.torch_rng = torch.Generator()
            self.torch_rng.manual_seed(2147483647)
            self.shuffle_idxs = torch.randperm(img_size**2, generator=self.torch_rng)

        def __getitem__(self, index):
            sample = super().__getitem__(index)
            image, label = sample

            if self.img_size > 28:
                new_image = torch.full((1, self.img_size, self.img_size), image.min())
                c_idx = self.c_idxs[index]
                r_idx = self.r_idxs[index]
                new_image[:, c_idx : c_idx + 28, r_idx : r_idx + 28] = image
                image = new_image

            if self.noise:
                self.torch_rng.manual_seed(2147433433 + index)
                image = image + self.noise * torch.randn(
                    image.shape, generator=self.torch_rng
                )

            if self.scramble_image:
                image = image.view(-1)[self.shuffle_idxs].reshape(
                    1, self.img_size, self.img_size
                )

            return (image, label)

    return ModifiedDataset


def get_dataloaders(
    base_dataset,
    batch_size,
    img_size=28,
    random_shift=False,
    scramble_image=False,
    noise=0.0,
    show_examples=False,
):
    dataset_cls = get_modified_dataset(base_dataset)
    if base_dataset == datasets.FashionMNIST:
        mean = 0.286041
        std = 0.353024
        labels_map = lambda label: {
            0: "T-Shirt",
            1: "Trouser",
            2: "Pullover",
            3: "Dress",
            4: "Coat",
            5: "Sandal",
            6: "Shirt",
            7: "Sneaker",
            8: "Bag",
            9: "Ankle Boot",
        }[label]
        path = "./data/FMNIST/"
    elif base_dataset == datasets.MNIST:
        mean = 0.1307
        std = 0.3081
        labels_map = lambda label: label
        path = "./data/MNIST/"
    else:
        raise NotImplementedError

    ### SOLUTION
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((mean,), (std,))]
    )
    train_set = dataset_cls(
        path,
        train=True,
        download=True,
        transform=transform,
        img_size=img_size,
        random_shift=random_shift,
        scramble_image=scramble_image,
        noise=noise,
    )
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,  # Shuffle the iteration order over the dataset
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        num_workers=2,
    )

    val_set = dataset_cls(
        path,
        train=False,
        download=True,
        transform=transform,
        img_size=img_size,
        random_shift=random_shift,
        scramble_image=scramble_image,
        noise=noise,
    )
    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
    )
    ### TEMPLATE
    # # ***************************************************
    # # INSERT YOUR CODE HERE
    # # TODO: Create the datasets and dataloaders with the
    # # right arguments
    # # ***************************************************
    # transform = transforms.Compose([
    # # TODO: Insert ToTensor and Normalize using mean and std.
    # # We can use the same transforms for train and val since we don't perform
    # # any augmentations.
    # ])
    # train_set = dataset_cls(
    # # TODO: Add the appropriate arguments
    # )
    # train_loader = torch.utils.data.DataLoader(
    # # TODO: Add the appropriate arguments
    # )
    # val_set = dataset_cls(
    # # TODO: Add the appropriate arguments
    # )
    # val_loader = torch.utils.data.DataLoader(
    # # TODO: Add the appropriate arguments
    # )
    # raise NotImplementedError
    ### END SOLUTION

    if show_examples:
        figure = plt.figure(figsize=(8, 8))
        figure.suptitle("Example Data")
        cols, rows = 3, 3
        for i in range(1, cols * rows + 1):
            sample_idx = torch.randint(len(train_set), size=(1,)).item()
            img, label = train_set[sample_idx]
            figure.add_subplot(rows, cols, i)
            plt.title(labels_map(label))
            plt.axis("off")
            plt.imshow(img.squeeze(), cmap="gray")
        plt.show()
    return train_loader, val_loader


# Small test and visualization of the data
temp = get_dataloaders(
    datasets.MNIST,
    batch_size=32,
    img_size=56,
    random_shift=True,
    scramble_image=False,
    noise=1.0,
    show_examples=True,
)
del temp

### SOLUTION
get_mlp = lambda image_size: torch.nn.Sequential(
    torch.nn.Flatten(),
    torch.nn.Linear(image_size * image_size, 256),
    torch.nn.ReLU(),
    torch.nn.Linear(256, 256),
    torch.nn.ReLU(),
    torch.nn.Linear(256, 256),
    torch.nn.ReLU(),
    torch.nn.Linear(256, 256),
    torch.nn.ReLU(),
    torch.nn.Linear(256, 10),
)

get_cnn = lambda image_size: torch.nn.Sequential(
    torch.nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2),
    torch.nn.ReLU(),
    torch.nn.Conv2d(32, 64, 5, stride=2, padding=2),
    torch.nn.ReLU(),
    torch.nn.Conv2d(64, 64, 5, stride=1, padding=2),
    torch.nn.ReLU(),
    torch.nn.Conv2d(64, 128, 5, stride=2, padding=2),
    torch.nn.ReLU(),
    torch.nn.AdaptiveAvgPool2d(1),
    torch.nn.Conv2d(128, 10, 1),
    torch.nn.Flatten(),
)
### TEMPLATE
# # ***************************************************
# # INSERT YOUR CODE HERE
# # ***************************************************
# get_mlp = lambda image_size: torch.nn.Sequential(
# # TODO: Insert the appropriate arguments here
# )

# get_cnn = lambda image_size: torch.nn.Sequential(
# # TODO: Insert the appropriate arguments here
# )
### END SOLUTION

def train_epoch(model, optimizer, scheduler, criterion, train_loader, epoch, device):
    ### SOLUTION
    model.train()
    loss_history = []
    accuracy_history = []
    lr_history = []
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        scheduler.step()

        pred = output.argmax(dim=1, keepdim=True)
        correct = pred.eq(target.view_as(pred)).sum().item()
        loss_float = loss.item()
        accuracy_float = correct / len(data)

        loss_history.append(loss_float)
        accuracy_history.append(accuracy_float)
        lr_history.append(scheduler.get_last_lr()[0])
        if batch_idx % (len(train_loader.dataset) // len(data) // 10) == 0:
            print(
                f"Train Epoch: {epoch}-{batch_idx:03d} "
                f"batch_loss={loss_float:0.2e} "
                f"batch_acc={accuracy_float:0.3f} "
                f"lr={scheduler.get_last_lr()[0]:0.3e} "
            )
    ### TEMPLATE
    # # ***************************************************
    # # TODO: FILL IN THE DETAILS BELOW
    # # ***************************************************
    # # TODO: Set model to training mode (affects dropout, batch norm e.g.)
    # loss_history = []
    # accuracy_history = []
    # lr_history = []
    # # TODO: Change the loop to get batch_idx, data and target from train_loader
    # for _ in something:
    #     # TODO: Move the data to the device
    #     # TODO: Zero the gradients
    #     # TODO: Compute model output
    #     # TODO: Compute loss
    #     # TODO: Backpropagate loss
    #     # TODO: Perform an optimizer step
    #     # TODO: Perform a learning rate scheduler step

    #     # TODO: Compute accuracy_float (float value, not a tensor)
    #     # TODO: Compute loss_float (float value, not a tensor)
    #     # TODO: Add loss_float to loss_history
    #     # TODO: Add accuracy_float to accuracy_history

    #     loss_history.append(loss_float)
    #     accuracy_history.append(accuracy_float)
    #     lr_history.append(scheduler.get_last_lr()[0])
    #     if batch_idx % (len(train_loader.dataset) // len(data) // 10) == 0:
    #         print(
    #             f"Train Epoch: {epoch}-{batch_idx:03d} "
    #             f"batch_loss={loss_float:0.2e} "
    #             f"batch_acc={accuracy_float:0.3f} "
    #             f"lr={scheduler.get_last_lr()[0]:0.3e} "
    #         )
    ### END SOLUTION

    return loss_history, accuracy_history, lr_history


@torch.no_grad()
def validate(model, device, val_loader, criterion):
    model.eval()  # Important: eval mode (affects dropout, batch norm etc)
    test_loss = 0
    correct = 0
    for data, target in val_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        test_loss += criterion(output, target).item() * len(data)
        pred = output.argmax(
            dim=1, keepdim=True
        )  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(val_loader.dataset)

    print(
        "Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)".format(
            test_loss,
            correct,
            len(val_loader.dataset),
            100.0 * correct / len(val_loader.dataset),
        )
    )
    return test_loss, correct / len(val_loader.dataset)


@torch.no_grad()
def get_predictions(model, device, val_loader, criterion, num=None):
    model.eval()
    points = []
    for data, target in val_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = criterion(output, target)
        pred = output.argmax(dim=1, keepdim=True)

        data = np.split(data.cpu().numpy(), len(data))
        loss = np.split(loss.cpu().numpy(), len(data))
        pred = np.split(pred.cpu().numpy(), len(data))
        target = np.split(target.cpu().numpy(), len(data))
        points.extend(zip(data, loss, pred, target))

        if num is not None and len(points) > num:
            break

    return points


def run_training(
    model_factory,
    num_epochs,
    optimizer_kwargs,
    data_kwargs,
    device="cuda",
):
    # ===== Data Loading =====
    train_loader, val_loader = get_dataloaders(**data_kwargs)

    # ===== Model, Optimizer and Criterion =====
    model = model_factory()
    model = model.to(device=device)
    optimizer = torch.optim.AdamW(model.parameters(), **optimizer_kwargs)
    criterion = torch.nn.functional.cross_entropy
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=(len(train_loader.dataset) * num_epochs) // train_loader.batch_size,
    )

    # ===== Train Model =====
    lr_history = []
    train_loss_history = []
    train_acc_history = []
    val_loss_history = []
    val_acc_history = []
    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc, lrs = train_epoch(
            model, optimizer, scheduler, criterion, train_loader, epoch, device
        )
        train_loss_history.extend(train_loss)
        train_acc_history.extend(train_acc)
        lr_history.extend(lrs)

        val_loss, val_acc = validate(model, device, val_loader, criterion)
        val_loss_history.append(val_loss)
        val_acc_history.append(val_acc)

    # ===== Plot training curves =====
    n_train = len(train_acc_history)
    t_train = num_epochs * np.arange(n_train) / n_train
    t_val = np.arange(1, num_epochs + 1)

    plt.figure(figsize=(6.4 * 3, 4.8))
    plt.subplot(1, 3, 1)
    plt.plot(t_train, train_acc_history, label="Train")
    plt.plot(t_val, val_acc_history, label="Val")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")

    plt.subplot(1, 3, 2)
    plt.plot(t_train, train_loss_history, label="Train")
    plt.plot(t_val, val_loss_history, label="Val")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.subplot(1, 3, 3)
    plt.plot(t_train, lr_history)
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")

    # ===== Plot low/high loss predictions on validation set =====
    points = get_predictions(
        model,
        device,
        val_loader,
        partial(torch.nn.functional.cross_entropy, reduction="none"),
    )
    points.sort(key=lambda x: x[1])
    plt.figure(figsize=(15, 6))
    for k in range(5):
        plt.subplot(2, 5, k + 1)
        plt.imshow(points[k][0][0, 0], cmap="gray")
        plt.title(f"true={int(points[k][3])} pred={int(points[k][2])}")
        plt.subplot(2, 5, 5 + k + 1)
        plt.imshow(points[-k - 1][0][0, 0], cmap="gray")
        plt.title(f"true={int(points[-k-1][3])} pred={int(points[-k-1][2])}")

    return sum(train_acc) / len(train_acc), val_acc

### SOLUTION
# Running the three configurations for the MLP we got the following accuracies
# (expect these to vary slightly between runs):
# 9355/10000, 5738/10000, 9363/10000
# For the CNN we got:
# 9593/10000, 9565/10000, 6931/10000
# We observe that:
# The MLP is strongly affected by the shifts
# The MLP is not affected by the scrambling
# The CNN is only slightly affect by the shifts
# The CNN is significantly affected by the scrambling (but still learns)
### TEMPLATE
### END SOLUTION

image_size = 56
model_factory = lambda: get_mlp(image_size)
num_epochs = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

optimizer_kwargs = dict(
    lr=1e-3,
    weight_decay=1e-2,
)
data_kwargs = dict(
    base_dataset=datasets.MNIST,
    batch_size=128,
    img_size=image_size,
    random_shift=False,
    scramble_image=False,
    noise=1.0,
    show_examples=True,
)

run_training(
    model_factory=model_factory,
    num_epochs=num_epochs,
    optimizer_kwargs=optimizer_kwargs,
    data_kwargs=data_kwargs,
    device=device,
)