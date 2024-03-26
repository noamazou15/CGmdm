import torch
class Square:
    def __init__(self, center_x=0, center_z=0, size=0.5, height=0.1, color=(1, 0, 0, 0.5)):
        # Initialize the square's properties
        self.center_x = center_x
        self.center_z = center_z
        self.size = size
        self.height = height
        self.color = color

    def get_center_x(self):
        return self.center_x

    def get_center_z(self):
        return self.center_z

    def get_size(self):
        return self.size

    def get_height(self):
        return self.height

    def get_color(self):
        return self.color

    def get_boundaries(self):
        # Calculate the boundaries
        min_x = self.center_x - self.size
        max_x = self.center_x + self.size
        min_z = self.center_z - self.size
        max_z = self.center_z + self.size
        min_y = self.height  # Ensure the square is plotted above the base plane
        return torch.tensor([min_x, max_x, min_z, max_z], dtype=torch.float, device="cuda:0")
    
    def is_in_square(self, traj):
        square_boundaries = self.get_boundaries()
        is_inside_x = (traj[:, 0] > square_boundaries[0]) & (traj[:, 0] < square_boundaries[1])
        is_inside_z = (traj[:, 1] > square_boundaries[2]) & (traj[:, 1] < square_boundaries[3])
        return is_inside_x & is_inside_z
    
    def dist_from_square(self, traj):
        square_boundaries = self.get_boundaries()
        x_dist_min = torch.min(torch.abs(traj[:, 0] - square_boundaries[0]), torch.abs(traj[:, 0] - square_boundaries[1]))
        z_dist_min = torch.min(torch.abs(traj[:, 1] - square_boundaries[2]), torch.abs(traj[:, 1] - square_boundaries[3]))
        dist = torch.sqrt(x_dist_min**2 + z_dist_min**2)
        return dist

class SquareList:
    def __init__(self, squares):
        self.squares = squares
    
    def is_in_all_squares(self, traj):
        mask = torch.zeros(traj.shape[0], dtype=torch.bool, device="cuda:0")
        for square in self.squares:
            mask |= square.is_in_square(traj)
        return mask
    
    def min_dist_from_all_squares(self, traj):
        dist = torch.tensor(float("inf"), dtype=torch.float, device="cuda:0")
        for square in self.squares:
            dist = torch.min(dist, square.dist_from_square(traj))
        return dist

    def __iter__(self):
        return iter(self.squares)
    
# #create 10 sqaures in two different rows and 5 columns
squares_list = []
# for j in [-2, 2]:
#     for i in [0,2, 4, 6, 8, 10]:
#         squares.append(Square(center_x=j, center_z=i, size=0.5, height=0.1, color=(1, 0, 0, 0.5)))

# #add another two rows
# for j in [-1, 1.]:
#     for i in [1, 3, 5, 7, 9]:
#         squares.append(Square(center_x=j, center_z=i, size=0.5, height=0.1, color=(1, 0, 0, 0.5)))

#draw a diagonal of squares
for i in range(6):
    squares_list.append(Square(center_x=i-2, center_z=i+2, size=0.5, height=0.1, color=(1, 0, 0, 0.5)))

squares = SquareList(squares_list)

# for i in range(5):
#     squares.append(Square(center_x=0, center_z=i, size=0.5, height=0.1, color=(1, 0, 0, 0.5)))

# # Horizontal part of the L (this will start right where the vertical part ends)
# for i in range(1, 5 + 1):  # Start from 1 to avoid overlapping with the vertical arm
#     squares.append(Square(center_x=i, center_z=5 - 1, size=0.5, height=0.1, color=(1, 0, 0, 0.5)))