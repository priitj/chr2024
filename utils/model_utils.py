import os
import time
from tempfile import TemporaryDirectory
import torch


def train_model(model, criterion, optimizer, scheduler, loaders, device, dataset_sizes, num_epochs=25):
    since = time.time()

    # Create a temporary directory to save training checkpoints
    with TemporaryDirectory() as tempdir:
        best_model_params_path = os.path.join(tempdir, 'best_model_params.pt')
        torch.save(model.state_dict(), best_model_params_path)
        best_acc = 0.0

        for epoch in range(num_epochs):
            print(f'Epoch {epoch}/{num_epochs - 1}')
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()  # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for inputs, labels in loaders[phase]:
                    inputs, labels = inputs.to(device), labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                if phase == 'train' and scheduler is not None:
                    scheduler.step()

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.to(torch.float32) / dataset_sizes[phase]

                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save(model.state_dict(), best_model_params_path)

            print()

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {best_acc:4f}')
        print()
        print('~' * 10)
        print()

        # load best model weights
        model.load_state_dict(torch.load(best_model_params_path))
    return model


def test_model(model, criterion, class_names, loaders, device):
    # track test loss
    test_loss = 0.0
    class_correct = {key: 0 for key in range(len(class_names))}
    class_prediction_incorrect = {key: 0 for key in range(len(class_names))}
    class_total = {key: 0 for key in range(len(class_names))}

    model.eval()
    # iterate over test data; each iteration below returns a batch of features and labels
    for data, labels in loaders['test']:
        data, labels = data.to(device), labels.to(device)
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the batch loss
        loss = criterion(output, labels)
        # update test loss
        test_loss += loss.item() * data.size(0)
        # convert output probabilities to a list of predicted class indices
        _, pred = torch.max(output, 1)
        # calculate test accuracy for each object class
        for i in range(len(pred)):
            true_class_idx = labels[i].item()
            predicted_class_idx = pred[i].item()
            class_correct[true_class_idx] += (true_class_idx == predicted_class_idx)
            class_prediction_incorrect[predicted_class_idx] += (predicted_class_idx != true_class_idx)
            class_total[true_class_idx] += 1

    # average test loss
    test_loss = test_loss / len(loaders['test'].dataset)
    print('Test Loss: {:.6f}\n'.format(test_loss))

    for i in range(0, len(class_names)):
        if class_total[i] > 0:
            print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
                class_names[i], 100 * class_correct[i] / class_total[i],
                class_correct[i], class_total[i]))
        else:
            print('Test Accuracy of %5s: N/A (no training examples)' % (class_names[i]))

    print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
        100. * sum(class_correct.values()) / sum(class_total.values()),
        sum(class_correct.values()), sum(class_total.values())))

    return [class_correct, class_prediction_incorrect, class_total]

def predict(model, class_names, loader, device):
    model.eval()
    # iterate over test data; each iteration below returns a batch of features and labels
    all_preds = []
    all_labels = []
    all_output = []
    for data, labels in loader:
        data = data.to(device)
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        _, pred = torch.max(output, 1)
        all_preds += pred.detach().cpu().numpy().tolist()
        all_output += output.detach().cpu().numpy().tolist()
        all_labels += labels.cpu().numpy().tolist()

    return all_preds, all_labels, all_output

def preds_to_dict(class_names, preds, labels, output):
    # convert to the format used by the CLIP classifier
    predictions = []
    for i, c in enumerate(preds):
        predictions.append({
            "image": 0,  # loader doesn't give the filename? dummy filler
            "true_class": class_names[labels[i]],
            "predicted_class": class_names[c],
            "correct": labels[i] == c,
            "predictions": output[i]
        })
    return predictions

