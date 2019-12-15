import json
import matplotlib
matplotlib.use('Agg')  # This is hacky (useful for running on VMs)
import numpy as np
import os
import time
import torch
from anode.models import ODENet
from anode.conv_models import ConvODENet
from anode.discrete_models import ResNet
from anode.training import Trainer
from experiments.dataloaders import mnist, cifar10, tiny_imagenet


def run_and_save_experiments_img(device, path_to_config):
    """Runs and saves experiments as they are produced (so results are still
    saved even if NFEs become excessively large or underflow occurs).

    Parameters
    ----------
    device : torch.device

    path_to_config : string
        Path to config json file.
    """
    # Open config file
    with open(path_to_config) as config_file:
        config = json.load(config_file)

    # Create a folder to store experiment results
    timestamp = time.strftime("%Y-%m-%d_%H-%M")
    directory = "img_results_{}_{}".format(timestamp, config["id"])
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Save config file in experiment directory
    with open(directory + '/config.json', 'w') as config_file:
        json.dump(config, config_file)

    num_reps = config["num_reps"]
    dataset = config["dataset"]
    model_configs = config["model_configs"]
    training_config = config["training_config"]
    noise_configs = config["noise_configs"]

    results = {"dataset": dataset, "model_info": []}

    if dataset == 'mnist':
        data_loader, test_loader = mnist(training_config["batch_size"])
        img_size = (1, 28, 28)
        output_dim = 10

    if dataset == 'cifar10':
        data_loader, test_loader = cifar10(training_config["batch_size"])
        img_size = (3, 32, 32)
        output_dim = 10

    if dataset == 'imagenet':
        data_loader = tiny_imagenet(training_config["batch_size"])
        img_size = (3, 64, 64)
        output_dim = 200

    only_success = True  # Boolean to keep track of any experiments failing

    for i, model_config in enumerate(model_configs):
        results["model_info"].append({})
        # Keep track of losses
        epoch_loss_histories = []
        # Keep track of models potentially failing
        model_stats = {
            "exceeded": {"count": 0, "final_losses": []},
            "underflow": {"count": 0, "final_losses": []},
            "success": {"count": 0, "final_losses": []}
        }

        epoch_loss_val_histories = {
            "none": []
        }
        epoch_acc_val_histories = {
            "none": []
        }
        for noise_config in noise_configs:
            noise_type = noise_config["type"] + "%.1f" % noise_config["param"]
            epoch_loss_val_histories[noise_type] = []
            epoch_acc_val_histories[noise_type] = []

        is_ode = model_config["type"] == "odenet" or model_config["type"] == "anode"

        for j in range(num_reps):
            print("{}/{} model, {}/{} rep".format(i + 1, len(model_configs), j + 1, num_reps))

            if is_ode:
                if model_config["type"] == "odenet":
                    augment_dim = 0
                else:
                    augment_dim = model_config["augment_dim"]

                model = ConvODENet(device, img_size, model_config["num_filters"],
                                   output_dim=output_dim,
                                   augment_dim=augment_dim,
                                   time_dependent=model_config["time_dependent"],
                                   non_linearity=model_config["non_linearity"],
                                   adjoint=True)
            else:
                model = ResNet(data_dim, model_config["hidden_dim"],
                               model_config["num_layers"],
                               output_dim=output_dim,
                               is_img=True)

            model.to(device)

            optimizer = torch.optim.Adam(model.parameters(),
                                         lr=model_config["lr"],
                                         weight_decay=model_config["weight_decay"])

            trainer = Trainer(model, optimizer, device,
                              classification=True,
                              print_freq=training_config["print_freq"],
                              record_freq=training_config["record_freq"],
                              verbose=True,
                              save_dir=(directory, '{}_{}'.format(i, j)))

            epoch_loss_histories.append([])

            # Train one epoch at a time, as NODEs can underflow or exceed the
            # maximum NFEs
            for epoch in range(training_config["epochs"]):
                print("\nEpoch {}".format(epoch + 1))
                try:
                    trainer.train(data_loader, 1)
                    end_training = False
                except AssertionError as e:
                    only_success = False
                    # Assertion error means we either underflowed or exceeded
                    # the maximum number of steps
                    error_message = e.args[0]
                    # Error message in torchdiffeq for max_num_steps starts
                    # with 'max_num_steps'
                    if error_message.startswith("max_num_steps"):
                        print("Maximum number of steps exceeded")
                        file_name_root = 'exceeded'
                    elif error_message.startswith("underflow"):
                        print("Underflow")
                        file_name_root = 'underflow'
                    else:
                        print("Unknown assertion error")
                        file_name_root = 'unknown'

                    model_stats[file_name_root]["count"] += 1

                    if len(trainer.buffer['loss']):
                        final_loss = np.mean(trainer.buffer['loss'])
                    else:
                        final_loss = None
                    model_stats[file_name_root]["final_losses"].append(final_loss)

                    end_training = True

                # Save info at every epoch
                epoch_loss_histories[-1] = trainer.histories['epoch_loss_history']

                epoch_loss_val = dataset_mean_loss(trainer, test_loader, device, "none")
                epoch_loss_val_histories["none"].append(epoch_loss_val)
                epoch_acc_val = dataset_acc(trainer, test_loader, device, "none")
                epoch_acc_val_histories["none"].append(epoch_acc_val)
                print("none: loss: {:.3f} acc: {:.3f}".format(epoch_loss_val, epoch_acc_val))

                for noise_config in noise_configs:
                    noise_type = noise_config["type"] + "%.1f" % noise_config["param"]
                    epoch_loss_val = dataset_mean_loss(trainer, test_loader, device, noise_config["type"], noise_config["param"])
                    epoch_loss_val_histories[noise_type].append(epoch_loss_val)
                    epoch_acc_val = dataset_acc(trainer, test_loader, device, noise_config["type"], noise_config["param"])
                    epoch_acc_val_histories[noise_type].append(epoch_acc_val)
                    print("{}: loss: {:.3f} acc: {:.3f}".format(noise_type, epoch_loss_val, epoch_acc_val))

                results["model_info"][-1]["type"] = model_config["type"]
                results["model_info"][-1]["epoch_loss_history"] = epoch_loss_histories
                results["model_info"][-1]["epoch_loss_val_history"] = epoch_loss_val_histories
                results["model_info"][-1]["epoch_acc_val_history"] = epoch_acc_val_histories

                # Save losses and nfes at every epoch
                with open(directory + '/losses.json', 'w') as f:
                    json.dump(results['model_info'], f)

                # If training failed, move on to next rep
                if end_training:
                    break

                # If we reached end of training, increment success counter
                if epoch == training_config["epochs"] - 1:
                    model_stats["success"]["count"] += 1

                    if len(trainer.buffer['loss']):
                        final_loss = np.mean(trainer.buffer['loss'])
                    else:
                        final_loss = None
                    model_stats["success"]["final_losses"].append(final_loss)

        # Save model stats
        with open(directory + '/model_stats{}.json'.format(i), 'w') as f:
            json.dump(model_stats, f)

def dataset_mean_loss(trainer, data_loader, device, noise_type="none", noise_param=0):
    """Returns mean loss of model on a dataset. Useful for calculating
    validation loss.

    Parameters
    ----------
    trainer : training.Trainer instance
        Trainer instance for model we want to evaluate.

    data_loader : torch.utils.data.DataLoader

    device : torch.device
    """
    epoch_loss = 0.
    for x_batch, y_batch in data_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        if noise_type != "none":
            if noise_type == "gauss":
                x_batch += gauss(x_batch, noise_param)
            elif noise_type == "fgsm":
                x_batch += fgsm(trainer, x_batch, y_batch, noise_param)
            elif noise_type == "pgd":
                x_batch += pgd(trainer, x_batch, y_batch, noise_param)
        y_pred = trainer.model(x_batch)
        loss = trainer.loss_func(y_pred, y_batch)
        epoch_loss += loss.item()
    return epoch_loss / len(data_loader)

def dataset_acc(trainer, data_loader, device, noise_type="none", noise_param=0):
    """Returns accuracy of model on a dataset. Useful for calculating
    validation accuracy.

    Parameters
    ----------
    trainer : training.Trainer instance
        Trainer instance for model we want to evaluate.

    data_loader : torch.utils.data.DataLoader

    device : torch.device
    """
    correct = 0
    total = 0
    for x_batch, y_batch in data_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        if noise_type != "none":
            if noise_type == "gauss":
                x_batch += gauss(x_batch, noise_param)
            elif noise_type == "fgsm":
                x_batch += fgsm(trainer, x_batch, y_batch, noise_param)
            elif noise_type == "pgd":
                x_batch += pgd(trainer, x_batch, y_batch, noise_param)
        _, y_pred = torch.max(trainer.model(x_batch), 1)
        correct += (y_pred == y_batch).sum().item()
        total += y_batch.size(0)
    return correct / total

def gauss(x, std):
    """Perturb by gaussian noise with given standard deviation."""
    return torch.randn(x.size()) * std 

def fgsm(trainer, x, y, epsilon):
    """FGSM attack perturbs in the direction of the gradient."""
    delta = torch.zeros_like(x, requires_grad=True)
    loss = trainer.loss_func(trainer.model(x+delta), y)
    loss.backward()
    return epsilon * delta.grad.detach().sign()

def pgd(trainer, x, y, epsilon, alpha=1e-2, n=40):
    """PGD attack iterates FGSM."""
    delta = torch.zeros_like(x, requires_grad=True)
    for i in range(n):
        loss = trainer.loss_func(trainer.model(x+delta), y)
        loss.backward()
        delta.data = (delta + alpha * delta.grad.detach().sign()).clamp(-epsilon, epsilon)
        delta.grad.zero_()
    return delta.detach()
