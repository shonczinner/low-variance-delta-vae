from torchvision import transforms, datasets

def load_mnist_tensor(n_samples=None):

    transform = transforms.ToTensor()

    dataset = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=transform
    )

    x = dataset.data.float() / 255.0
    x = x.unsqueeze(1)

    y = dataset.targets

    if n_samples is not None:
        x = x[:n_samples]
        y = y[:n_samples]

    return x, y