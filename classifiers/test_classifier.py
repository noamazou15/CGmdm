import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from data_loaders.humanml_utils import HML_ROOT_MASK
from data_loaders.get_data import get_dataset_loader



input_size = 263 * 196  # Adjusted based on the shape of x
hidden_size1 = 256  # Adjust the size of the first hidden layer as needed
hidden_size2 = 512  # Adjust the size of the second hidden layer as needed

def test_cond(x, t, p_mean_var, **model_kwargs):
    x = x.to('cuda:0')
    circles = torch.load('./classifiers/circles.pt', map_location=torch.device('cuda:0'))
    #mask circles with the root mask
    HML_ROOT_MASK_tensor = torch.tensor(HML_ROOT_MASK, dtype=torch.bool, device=circles.device).unsqueeze(1).unsqueeze(2)
    circles_root = torch.zeros_like(circles)
    circles_root[0] = circles[0] * HML_ROOT_MASK_tensor # Forward pass
    

    with torch.enable_grad():
        x = x.detach().requires_grad_(True)
        masked_x = x * HML_ROOT_MASK_tensor
        loss = F.l1_loss(masked_x, circles_root)
        loss = loss.sum()
        grad = torch.autograd.grad(-loss, x)[0]
        return grad * 100

        
def test_cond_with_trained_classifier(x, t, p_mean_var, **model_kwargs):
    #load the classifier from the path
    classifier = Classifier(input_size, hidden_size1)
    classifier.load_state_dict(torch.load('l1_with_circles_root.pth'))
    classifier.to('cuda:0')
    with torch.enable_grad():
        x_in = x.detach().requires_grad_(True)
        logits = classifier(x_in)
        log_probs = F.log_softmax(logits, dim=-1)
        selected = log_probs
        gradients = torch.autograd.grad(selected.sum(), x_in)[0] 
    return gradients




class Classifier(nn.Module):
    def __init__(self, input_size, hidden_size1):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.relu1 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size1, input_size)

    def forward(self, x):
        original_shape = x.size()
        x = x.reshape(x.size(0), -1)  # Flatten the input
        x = self.relu1(self.fc1(x))
        x = self.fc3(x)
        return x.view(original_shape)


def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Instantiate the classifier
    classifier = Classifier(input_size, hidden_size1)
    classifier.to('cuda:0')

    # Define the L1 loss function : measure the difference between 
    def criterion(outputs, targets):
        outputs_trajectory = outputs * HML_ROOT_MASK_tensor
        loss_on_trajectory = F.l1_loss(outputs_trajectory, targets)
        return loss_on_trajectory
    
    # Define an optimizer (e.g., Stochastic Gradient Descent - SGD)
    optimizer = optim.SGD(classifier.parameters(), lr=0.1)


    circles = torch.load('./classifiers/circles.pt', map_location=torch.device('cuda:0'))
    #mask circles with the root mask
    HML_ROOT_MASK_tensor = torch.tensor(HML_ROOT_MASK, dtype=torch.bool, device=circles.device).unsqueeze(1).unsqueeze(2)
    circles_root = torch.zeros_like(circles)
    circles_root[0] = circles[0] * HML_ROOT_MASK_tensor # Forward pass
    y = circles_root
    data = get_dataset_loader(name="humanml",
                                  batch_size=1,
                                  num_frames=196,
                                  split='test',
                                  hml_mode='eval')
    for i,(x,cond) in enumerate(data):
        x = x.to('cuda:0')
        outputs = classifier(x)

        # Compute the L1 loss between x and y
        
        loss = criterion(outputs, y)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print the loss for monitoring progress
        print(f'Loss: {loss.item()}')

        
    torch.save(classifier.state_dict(), 'l1_with_circles_root.pth')

    

if __name__ == "__main__":
    main()