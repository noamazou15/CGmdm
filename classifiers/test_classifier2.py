import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from data_loaders.humanml_utils import HML_ROOT_MASK
from data_loaders.get_data import get_dataset_loader


input_size = 263 * 196 
hidden_size1 = 256
hidden_size2 = 512 

// Testing comment
class Classifier(nn.Module):
    def __init__(self, input_size, hidden_size1):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size1, 1)  # Output a single value representing the "safety" score

    def forward(self, x):
        x = x.reshape(x.size(0), -1)  # Flatten the input
        x = self.relu1(self.fc1(x))
        x = self.fc2(x)
        return x

def square_avoidance_loss(hip_joint_positions, square_boundaries, inside_penalty=1000):
    """
    Calculate a loss that penalizes the hip joint for being within the red square, 
    with a high penalty for being inside, and a scaled penalty based on distance when outside.
    
    :param hip_joint_positions: Tensor containing the positions of the hip joint.
    :param square_boundaries: Tensor containing the boundaries of the square [x_min, x_max, z_min, z_max].
    :param inside_penalty: The penalty for being inside the square.
    :return: The calculated loss.
    """
    # Check if hip joint is inside the square
    is_inside_x = (hip_joint_positions[:, 0] > square_boundaries[0]) & (hip_joint_positions[:, 0] < square_boundaries[1])
    is_inside_z = (hip_joint_positions[:, 2] > square_boundaries[2]) & (hip_joint_positions[:, 2] < square_boundaries[3])
    is_inside_square = is_inside_x & is_inside_z

    # Apply high penalty for being inside the square
    inside_square_penalty = torch.where(is_inside_square, inside_penalty, torch.tensor(0., device=hip_joint_positions.device))

    # Calculate distance to the nearest square boundary for positions outside the square
    x_dist_min = torch.where(is_inside_x, torch.tensor(0., device=hip_joint_positions.device), torch.min(torch.abs(hip_joint_positions[:, 0] - square_boundaries[0]), torch.abs(hip_joint_positions[:, 0] - square_boundaries[1])))
    z_dist_min = torch.where(is_inside_z, torch.tensor(0., device=hip_joint_positions.device), torch.min(torch.abs(hip_joint_positions[:, 2] - square_boundaries[2]), torch.abs(hip_joint_positions[:, 2] - square_boundaries[3])))

    # Combine distance penalties for positions outside the square
    # Inverse scaling for distance penalty: smaller distances incur greater penalties
    # Adding a small epsilon to prevent division by zero
    epsilon = 1e-6
    distance_penalty = 1 / (x_dist_min + z_dist_min + epsilon)

    # Combine penalties
    total_penalty = inside_square_penalty + distance_penalty

    # The loss is the sum of all penalties
    loss = total_penalty.sum()
    
    return loss


def compute_square_boundaries(hip_joint_position, square_size=0.5):
    """
    Compute the boundaries of the red square based on the current hip joint position.
    
    :param hip_joint_position: Current position of the hip joint.
    :param square_size: Half the side length of the square.
    :return: Tensor containing the boundaries of the square [x_min, x_max, z_min, z_max].
    """
    square_center_x = hip_joint_position[0]
    square_center_z = hip_joint_position[2]
    
    x_min = square_center_x - square_size
    x_max = square_center_x + square_size
    z_min = square_center_z - square_size
    z_max = square_center_z + square_size
    
    return torch.tensor([x_min, x_max, z_min, z_max], dtype=torch.float, device=hip_joint_position.device)


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

    # Load the circles pattern and the HML_ROOT_MASK
    circles = torch.load('./classifiers/circles.pt', map_location=torch.device('cuda:0'))
    HML_ROOT_MASK_tensor = torch.tensor(HML_ROOT_MASK, dtype=torch.bool, device=circles.device).unsqueeze(1).unsqueeze(2)
    circles_root = torch.zeros_like(circles)
    circles_root[0] = circles[0] * HML_ROOT_MASK_tensor # Forward pass
    y = circles_root
    
    # Define the red square's boundaries
    square_boundaries = torch.tensor([x_min, x_max, z_min, z_max], dtype=torch.float, device='cuda:0')

    data_loader = get_dataset_loader(name="humanml", 
                                     batch_size=1, 
                                     num_frames=196, 
                                     split='test', 
                                     hml_mode='eval')
    
    for i, (x, _) in enumerate(data_loader):
        x = x.to('cuda:0')
        outputs = classifier(x)
        
        # Assuming outputs give the hip joint positions, and you have a mechanism to extract
        # the current hip joint position from outputs similar to how 'trajec[index]' is used.
        current_hip_joint_position = outputs[0, :3]  # Simplified example; adjust as needed
        
        # Compute the red square's boundaries dynamically
        square_boundaries = compute_square_boundaries(current_hip_joint_position)
        
        # Compute L1 loss
        l1_loss = criterion(outputs, y)
        
        # Compute square avoidance loss
        avoidance_loss = square_avoidance_loss(outputs, square_boundaries, inside_penalty=1000)
        
        # Combine the L1 loss and the square avoidance loss
        total_loss = l1_loss + avoidance_loss
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        print(f'Iteration: {i}, Loss: {total_loss.item()}')


    torch.save(classifier.state_dict(), 'classifier_avoiding_square.pth')

if __name__ == "__main__":
    main()
