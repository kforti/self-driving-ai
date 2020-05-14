import torchvision
import torch

def load_model(path):
    model = torchvision.models.resnet50(pretrained=True)

    # Freeze model weights
    for param in model.parameters():
        param.requires_grad = False

    model.fc = torch.nn.Sequential(
        torch.nn.Linear(
            in_features=2048,
            out_features=1
        ),
    )
    state = torch.load(path)
    model.load_state_dict(state)
    model.eval()
    return model


transform_pipe = torchvision.transforms.Compose([
    torchvision.transforms.ToPILImage(),  # Convert np array to PILImage

    # Resize image to 224 x 224 as required by most vision models
    torchvision.transforms.Resize(
        size=(224, 224)
    ),

    # Convert PIL image to tensor with image values in [0, 1]
    torchvision.transforms.ToTensor(),

    torchvision.transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
