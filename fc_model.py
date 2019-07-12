import torch
from torch import nn
from torch import optim
import torchvision.models as models


def build_model(arch="vgg13", hidden_units=512):
    assert arch[:3] == "vgg"
    image_classification_model = getattr(models, arch)
    model = image_classification_model(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    if hidden_units > 0:
        model.classifier = nn.Sequential(
            nn.Linear(25088, hidden_units),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(hidden_units, 102),
            nn.LogSoftmax(dim=1)
        )
    else:
        model.classifier = nn.Sequential(
            nn.Linear(25088, 2),
            nn.LogSoftmax(dim=1)
        )

    return model


def load_checkpoint(filepath, device="cpu"):
    checkpoint = torch.load(filepath, map_location='cpu')

    model = build_model(arch="vgg16", hidden_units=0)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    model.to(device)
    model.eval()

    optimizer = optim.Adam(model.classifier.parameters(), lr=0.03)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    epoch = checkpoint['epoch']
    loss = checkpoint['loss']

    return model, optimizer, epoch, loss


def save_checkpoint(save_dir, model, optimizer, epoch, loss):
    model.to("cpu")
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'class_to_idx': model.class_to_idx,
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'loss': loss
    }
    torch.save(checkpoint, save_dir)


def validate(dataloader, model, criterion, device="cpu"):
    model.to(device)

    # Disable dropouts before evaluation
    model.eval()

    running_loss = 0
    accuracy = 0
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        # Disable gradient calculations when evaluating
        with torch.no_grad():
            # Calculate loss
            logps = model(images)
            running_loss += criterion(logps, labels)

            # Calculate accuracy
            ps = torch.exp(logps)
            top_ps, top_class = ps.topk(1, dim=1)
            matches = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(matches.type(torch.FloatTensor))

    # Enable dropouts after evaluation
    model.train()

    loss = running_loss / len(dataloader)
    accuracy = accuracy / len(dataloader)
    return loss, accuracy


def train(model, criterion, optimizer, trainloader, validloader, epochs=5, training_loss=0, device="cpu",
          save_dir=None):
    with active_session():
        model.to(device)

        print_metrics_every = 1
        steps = 0
        running_loss = training_loss
        for epoch in range(epochs):
            for images, labels in trainloader:
                steps += 1
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()

                logps = model(images)
                loss = criterion(logps, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                if steps % print_metrics_every == 0:
                    training_loss = running_loss / print_metrics_every
                    validation_loss, validation_accuracy = validate(validloader, model, criterion, device=device)

                    print("Epoch: {}/{}...".format(epoch + 1, epochs),
                          "Training loss: {:.3f}...".format(training_loss),
                          "Validation loss: {:.3f}...".format(validation_loss),
                          "Validation Accuracy: {:.3f}...".format(validation_accuracy))

                    if save_dir:
                        save_checkpoint(save_dir, model, optimizer, epoch, training_loss)

                    running_loss = 0
