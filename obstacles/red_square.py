square_center_x = 0
square_center_z = 0

square_size = 0.5  
square_height = 0.1

red_square_color = (1, 0, 0, 0.5) # Color red

def get_center_x():
    return square_center_x

def get_center_z():
    return square_center_z

def get_size():
    return square_size

def get_height():
    return square_height

def get_color():
    return red_square_color

def get_boundaries(square_center_x=square_center_x,
                              square_center_z=square_center_z,
                              square_size=square_size):
    # Calculate the boundaries
    red_square_min_x = square_center_x - square_size
    red_square_max_x = square_center_x + square_size
    red_square_min_z = square_center_z - square_size
    red_square_max_z = square_center_z + square_size
    miny = square_height  # Ensure the square is plotted above the base plane
    
    return red_square_min_x, red_square_max_x, miny, red_square_min_z, red_square_max_z
