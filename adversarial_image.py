import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import json
import urllib.request
import ssl
import certifi

# Fix SSL certificate verification
ssl._create_default_https_context = ssl._create_unverified_context

# Download ImageNet class labels
def get_imagenet_labels():
    url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
    with urllib.request.urlopen(url) as f:
        labels = json.load(f)
    return labels

# Load pretrained model
def load_model():
    model = models.resnet50(pretrained=True)
    model.eval()
    return model

# Preprocess and deprocess functions
def preprocess(img):
    """Convert PIL image to tensor"""
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    return transform(img).unsqueeze(0)

def deprocess(tensor):
    """Convert tensor back to displayable image"""
    tensor = tensor.squeeze(0).detach().cpu()
    # Denormalize
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    tensor = tensor * std + mean
    tensor = torch.clamp(tensor, 0, 1)
    return transforms.ToPILImage()(tensor)

def generate_adversarial_image(model, target_class, labels, iterations=300, lr=0.1):
    """
    Generate an image that the model classifies as target_class
    
    Args:
        model: Pretrained neural network
        target_class: Index of target class (0-999 for ImageNet)
        labels: List of class labels
        iterations: Number of optimization steps
        lr: Learning rate
    """
    # Start with random noise
    img = torch.randn(1, 3, 224, 224, requires_grad=True)
    
    # Target vector (one-hot encoded)
    target = torch.zeros(1, 1000)
    target[0, target_class] = 1.0
    
    # Optimizer
    optimizer = torch.optim.Adam([img], lr=lr)
    
    print(f"Generating image for class: {labels[target_class]}")
    print("-" * 50)
    
    for i in range(iterations):
        optimizer.zero_grad()
        
        # Forward pass
        output = model(img)
        probs = torch.nn.functional.softmax(output, dim=1)
        
        # Loss: mean squared error between output and target
        loss = torch.mean((probs - target) ** 2)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Print progress
        if (i + 1) % 50 == 0:
            pred_class = output.argmax(dim=1).item()
            confidence = probs[0, target_class].item()
            print(f"Iteration {i+1}/{iterations}")
            print(f"  Loss: {loss.item():.6f}")
            print(f"  Target class confidence: {confidence:.4f}")
            print(f"  Predicted class: {labels[pred_class]}")
            print()
    
    return img

def visualize_results(img, model, labels):
    """Display the generated image and top predictions"""
    # Get predictions
    with torch.no_grad():
        output = model(img)
        probs = torch.nn.functional.softmax(output, dim=1)
        top5_prob, top5_idx = torch.topk(probs, 5)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Display image
    display_img = deprocess(img)
    ax1.imshow(display_img)
    ax1.axis('off')
    ax1.set_title('Generated Image', fontsize=14, fontweight='bold')
    
    # Display predictions
    top5_labels = [labels[idx] for idx in top5_idx[0]]
    top5_probs = top5_prob[0].cpu().numpy()
    
    y_pos = np.arange(len(top5_labels))
    ax2.barh(y_pos, top5_probs, color='green', alpha=0.7)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(top5_labels)
    ax2.invert_yaxis()
    ax2.set_xlabel('Confidence', fontsize=12)
    ax2.set_title('Top 5 Predictions', fontsize=14, fontweight='bold')
    ax2.set_xlim([0, 1])
    
    for i, v in enumerate(top5_probs):
        ax2.text(v + 0.02, i, f'{v:.3f}', va='center')
    
    plt.tight_layout()
    plt.show()

# Main execution
if __name__ == "__main__":
    # Load model and labels
    print("Loading model and labels...")
    model = load_model()
    labels = get_imagenet_labels()
    
    # Choose target class
    # Some interesting classes:
    # 39: iguana
    # 281: tabby cat
    # 388: giant panda
    # 933: cheeseburger
    # 609: iPod
    
    target_class = 39  # iguana
    
    # Generate adversarial image
    img = generate_adversarial_image(model, target_class, labels, iterations=300, lr=0.1)
    
    # Visualize results
    visualize_results(img, model, labels)
    
    print("\nDone! Try changing target_class to generate different images.")
    print("Some fun classes to try:")
    print("  39: iguana")
    print("  281: tabby cat")
    print("  388: giant panda")
    print("  933: cheeseburger")
    print("  437: baseball")