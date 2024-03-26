import torch
class Square:
    def __init__(self, center_x=0, center_z=0, size=0.5, height=0.1, color=(1, 0, 0, 0.5)):
        # Initialize the square's properties
        self.center_x = center_x
        self.center_z = center_z
        self.size = size
        self.height = height
        self.color = color

        # Calculate the boundaries
        self.min_x = self.center_x - self.size
        self.max_x = self.center_x + self.size
        self.min_z = self.center_z - self.size
        self.max_z = self.center_z + self.size

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
        return torch.tensor([self.min_x, self.max_x, self.min_z, self.max_z], dtype=torch.float, device="cuda:0")
    
    def is_in_square(self, traj):
        square_boundaries = self.get_boundaries()
        is_inside_x = (traj[:, 0] > square_boundaries[0]) & (traj[:, 0] < square_boundaries[1])
        is_inside_z = (traj[:, 1] > square_boundaries[2]) & (traj[:, 1] < square_boundaries[3])
        return is_inside_x & is_inside_z
    
    def dist_from_square(self, traj):
        square_boundaries = self.get_boundaries()
        x_dist_min = torch.min(torch.abs(traj[:, 0] - square_boundaries[0]), torch.abs(traj[:, 0] - square_boundaries[1]))
        z_dist_min = torch.min(torch.abs(traj[:, 1] - square_boundaries[2]), torch.abs(traj[:, 1] - square_boundaries[3]))
        dist = torch.sqrt((x_dist_min**2) + (z_dist_min**2))
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
    
    def create_squares():
        squares = []
        size = 0.1
        height = 0.1
        color = (1, 0, 0, 0.5)  
        space_between_squares = 1.0 
        space_between_rows = 1.5  
        indent = 0.5  

        # First row of squares
        for i in range(4):
            center_x = i * (2 * size + space_between_squares) - 3 * size - 1.5 * space_between_squares
            center_z = 3.5  # Starting Z position for the first row
            squares.append(Square(center_x=center_x, center_z=center_z, size=size, height=height, color=color))

        # Second row of squares (with indent)
        for i in range(4):
            center_x = i * (2 * size + space_between_squares) - 3 * size - 1.5 * space_between_squares + indent
            center_z = 3.5 - size - space_between_rows
            squares.append(Square(center_x=center_x, center_z=center_z, size=size, height=height, color=color))

        return squares


squares_list = SquareList.create_squares()

# Draw a diagonal of squares
# for i in range(6):
#     squares.append(Square(center_x=i-2, center_z=i+2, size=0.5, height=0.1, color=(1, 0, 0, 0.5)))

# squares_list = SquareList(squares)
