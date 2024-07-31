import wandb
import random
import torch
import numpy as np
import matplotlib.pyplot as plt


def _log_plot_and_image():
    # Generate a plot
    x = np.linspace(0, 2 * np.pi, 100)
    y = np.sin(x)

    # Save the plot to a file
    plt.figure()  # Create a new figure to avoid re-using old plot
    plt.plot(x, y)
    plt.title("Sine Wave")
    plt.xlabel("x")
    plt.ylabel("sin(x)")
    plt.savefig('sine_wave.png')  # Save plot as an image file
    plt.close()  # Close the plot to free memory

    # Log the plot as an image file
    wandb.log({"sine_wave": wandb.Image('sine_wave.png')})

    # Generate an image
    image = np.random.rand(28, 28)
    wandb.log({"random_image": wandb.Image(image, caption="Random Image")})

def _log_artifact(model):
    # Every epoch, update the model
    torch.save(model.state_dict(), "model_linear_test.pth")

    # Log model artifact
    artifact = wandb.Artifact('model', type='model')
    artifact.add_file('model_linear_test.pth')
    wandb.log_artifact(artifact)

def _log_matrics(epoch, offset):
    acc = 1 - 2 ** -epoch - random.random() / epoch - offset
    loss = 2 ** -epoch + random.random() / epoch + offset

    # log metrics to wandb
    wandb.log({"acc": acc, "loss": loss})


def main():
    # Initialize a wandb run
    wandb.init(
        # set the wandb project where this run will be logged
        project="wandb-cifar-test",

        # track hyperparameters and run metadata
        config={
            "learning_rate": 0.02,
            "architecture": "CNN",
            "dataset": "CIFAR-100",
            "epochs": 10,
        }
    )

    # A test training model
    model = torch.nn.Linear(10, 2)

    # simulate training
    epochs = 10
    offset = random.random() / 5
    for epoch in range(2, epochs):
        _log_matrics(epoch, offset)
        _log_plot_and_image()

    # Log artifact at the end of the training
    _log_artifact(model)

    # wandb
    wandb.finish()


if __name__ == "__main__":
    main()
