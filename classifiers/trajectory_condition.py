import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import classifiers.red_square as red_square

from classifiers.red_square import squares_list
from data_loaders.humanml_utils import HML_ROOT_MASK
from data_loaders.humanml.scripts.motion_process import recover_from_ric

# python -m sample.generate --model_path ./save/humanml_trans_enc_512/model000200000.pt --text_prompt The\ person\ is\ walking --num_samples 1 --num_repetitions 1 --cond_fn square

def arg_to_func(cond_arg, dataset):
    if (cond_arg == 'trajectory'):
        return trajectory_cond
    elif (cond_arg == 'square'):
        square_cond_with_mean_var = lambda x,t, p_mean_var, **model_kwargs: square_cond(x, t, p_mean_var, torch.from_numpy(dataset.t2m_dataset.mean), 
                                                                torch.from_numpy(dataset.t2m_dataset.std), model_kwargs=model_kwargs)
        return square_cond_with_mean_var
    else:
        return None
    
def trajectory_cond(x, t, p_mean_var, **model_kwargs):
    x = x.to('cuda:0')
    circles = torch.load('./classifiers/circles.pt', map_location=torch.device('cuda:0'))
    #mask circles with the root mask
    HML_ROOT_MASK_tensor = torch.tensor(HML_ROOT_MASK, dtype=torch.bool, device=circles.device).unsqueeze(1).unsqueeze(2)
    circles_root = torch.zeros_like(circles)
    circles_root[0] = circles[0] * HML_ROOT_MASK_tensor # Forward pass
    

    with torch.enable_grad():
        x = x.detach().requires_grad_(True)
        #x = recover_from_ric(x,22)
        masked_x = x * HML_ROOT_MASK_tensor
        loss = F.mse_loss(masked_x, circles_root)  
        loss = loss.sum()
        grad = torch.autograd.grad(-loss, x)[0]
        return grad * 20000


def square_cond(x, t ,p_mean_var ,mean ,var ,**model_kwargs):
    # Move input tensor to GPU for faster computation
    x = x.to('cuda:0')
    mean = mean.to('cuda:0')
    var = var.to('cuda:0')

    # Create a boolean tensor mask from HML_ROOT_MASK
    HML_ROOT_MASK_tensor = torch.tensor(HML_ROOT_MASK, dtype=torch.bool, device='cuda:0').unsqueeze(1).unsqueeze(2)

    # Detach x from the current computation graph and enable gradient tracking
    x = x.detach().requires_grad_(True)

    # Apply root mask to x
    x = x * HML_ROOT_MASK_tensor

    # Permute dimensions of x to match expected input structure for 'recover_from_ric'
    x_permuted = x.permute(0, 2, 3, 1)
    x_permuted = x_permuted * var + mean

    # Recover joint positions in 3D space with 'recover_from_ric'
    x_ric = recover_from_ric(x_permuted.float(),22)
    
    # Extract trajectory, ignoring y axis
    traj = x_ric[0][0][:, 0, [0, 2]]

    # Obtain the boundaries of the square obstacle
    # square_boundaries = square.get_boundaries()

    # Compute the loss based on the trajectory's interaction with the square boundaries
    # loss = square_avoidance_loss(traj, square_boundaries)

    loss = 0
    for square in squares_list:
        # Obtain the boundaries of the square
        square_boundaries = square.get_boundaries()
        # Compute the loss based on the trajectory's interaction with the square boundaries
        loss += square_avoidance_loss(traj, square_boundaries)

    # Compute gradient
    grad = torch.autograd.grad(-loss, x)[0] * 0.05
    
    return grad

def square_loss(traj, inside_penalty=5):
    in_square = square.is_in_square(traj)
    inside_square_penalty = torch.where(in_square, inside_penalty, torch.tensor(0., device=traj.device))
    dist = square.dist_from_square(traj)
    loss = inside_square_penalty + dist
    return loss.sum()

# Works good with inside_penalty=100000000
def square_avoidance_loss(traj, square_boundaries, inside_penalty=100000):
    # Check if hip joint is inside the square
    is_inside_x = (traj[:, 0] > square_boundaries[0]) & (traj[:, 0] < square_boundaries[1])
    is_inside_z = (traj[:, 1] > square_boundaries[2]) & (traj[:, 1] < square_boundaries[3])
    is_inside_square = is_inside_x & is_inside_z

    # Apply high penalty for being inside the square
    inside_square_penalty = torch.where(is_inside_square, inside_penalty, torch.tensor(0., device=traj.device))

    # Calculate distance to the nearest square boundary for positions outside the square
    x_dist_min = torch.where(is_inside_x, torch.tensor(0., device=traj.device), torch.min(torch.abs(traj[:, 0] - square_boundaries[0]), torch.abs(traj[:, 0] - square_boundaries[1])))
    z_dist_min = torch.where(is_inside_z, torch.tensor(0., device=traj.device), torch.min(torch.abs(traj[:, 1] - square_boundaries[2]), torch.abs(traj[:, 1] - square_boundaries[3])))

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


def trapezoid_square_avoidance_loss(traj, square_boundaries, min_distance=1.0, inside_penalty=1000000):
    # Constants
    epsilon = 1e-6
    
    # Extract square boundaries
    x_min, x_max, z_min, z_max = square_boundaries
    
    # Compute distance to the square's boundaries
    dist_to_x_min = traj[:, 0] - x_min
    dist_to_x_max = x_max - traj[:, 0]
    dist_to_z_min = traj[:, 1] - z_min
    dist_to_z_max = z_max - traj[:, 1]
    
    # Calculate penalties for being within the trapezoidal region near the boundaries
    x_penalty_min = torch.maximum(min_distance - dist_to_x_min, torch.tensor(0.0, device=traj.device))
    x_penalty_max = torch.maximum(min_distance - dist_to_x_max, torch.tensor(0.0, device=traj.device))
    z_penalty_min = torch.maximum(min_distance - dist_to_z_min, torch.tensor(0.0, device=traj.device))
    z_penalty_max = torch.maximum(min_distance - dist_to_z_max, torch.tensor(0.0, device=traj.device))
    
    # Sum the penalties for all sides to get the total distance penalty
    distance_penalty = x_penalty_min + x_penalty_max + z_penalty_min + z_penalty_max
    
    # Check if the hip joint is inside the square and apply high penalty
    is_inside_square = (traj[:, 0] > x_min) & (traj[:, 0] < x_max) & (traj[:, 1] > z_min) & (traj[:, 1] < z_max)
    inside_square_penalty = torch.where(is_inside_square, inside_penalty, torch.tensor(0.0, device=traj.device))
    
    # Combine penalties
    # Apply inside square penalty directly; distance penalty scales with closeness, but not inside the square
    total_penalty = torch.where(is_inside_square, inside_square_penalty, distance_penalty)
    
    # The loss is the sum of all penalties
    loss = total_penalty.sum()
    
    return loss


# def square_avoidance_loss(traj, square_boundaries, inside_penalty=1000, min_distance=1.0):
#     # Check if hip joint is inside the square
#     is_inside_x = (traj[:, 0] > square_boundaries[0]) & (traj[:, 0] < square_boundaries[1])
#     is_inside_z = (traj[:, 1] > square_boundaries[2]) & (traj[:, 1] < square_boundaries[3])
#     is_inside_square = is_inside_x & is_inside_z

#     # Apply high penalty for being inside the square
#     inside_square_penalty = torch.where(is_inside_square, inside_penalty, torch.tensor(0., device=traj.device))

#     # Calculate distance to the nearest square boundary for positions outside the square
#     x_dist_min = torch.where(is_inside_x, torch.tensor(0., device=traj.device),
#                              torch.min(torch.abs(traj[:, 0] - square_boundaries[0]),
#                                        torch.abs(traj[:, 0] - square_boundaries[1])))
#     z_dist_min = torch.where(is_inside_z, torch.tensor(0., device=traj.device),
#                              torch.min(torch.abs(traj[:, 1] - square_boundaries[2]),
#                                        torch.abs(traj[:, 1] - square_boundaries[3])))

#     # Determine if the distance to the boundary is below the minimum distance and calculate penalty
#     x_penalty = torch.where(x_dist_min <= min_distance, min_distance - x_dist_min, torch.tensor(0., device=traj.device))
#     z_penalty = torch.where(z_dist_min <= min_distance, min_distance - z_dist_min, torch.tensor(0., device=traj.device))

#     # Combine penalties
#     # When outside the min_distance, penalties are zero. Closer to square increases penalty linearly
#     distance_penalty = x_penalty + z_penalty

#     # Combine penalties
#     total_penalty = inside_square_penalty + distance_penalty

#     # The loss is the sum of all penalties
#     loss = total_penalty.sum()

#     return loss
