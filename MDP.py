import numpy as np 

# objectives list to get the targeted indexies
#      left   right     up     down
obj = [(0, -1),(0, 1),(-1, 0),(1, 0)]

dirs = ["Left", "Right", "Up", "down"]

# perpendicular list to get the noise direction indexies
#      left   right     up     down
perp = [(2, 3), (2, 3), (0, 1), (0, 1)]

# function that starts the game and initializes the map
def setup(
        dimentions : tuple,
        init_val : float = -0.04,
        gyms : list = None,
        walls : list = None,
        halls : list = None
          ) -> np.ndarray:
    m,n = dimentions
    grid = np.full((m*n), init_val).astype(np.float32)
    grid[gyms] = 1.0
    grid[walls] = -2.0
    grid[halls] = -1.0         
    return grid.reshape(m,n)

# function to validate if the interested position is valid or not
def dim_validation(
        grid : np.ndarray,
        i : int,
        j : int
        ) -> bool:
    m,n = grid.shape
    if 0 <= i <m and 0 <= j <n and grid[i][j] != -2.0:
        return True
    else:
        return False 

def Q_value(
        grid : np.ndarray,
        i : int,
        j : int,
        direction : int
        ) -> float:
    i_obj, j_obj = i+obj[direction][0] , j+obj[direction][1]
    if not dim_validation(grid, i_obj, j_obj):
        i_obj, j_obj = i, j

    i_p1, j_p1 = i+obj[perp[direction][0]][0], j+obj[perp[direction][0]][1]
    if not dim_validation(grid, i_p1, j_p1):
        i_p1, j_p1 = i, j

    i_p2, j_p2 = i+obj[perp[direction][1]][0], j+obj[perp[direction][1]][1]
    if not dim_validation(grid, i_p2, j_p2):
        i_p2, j_p2 = i, j

    q_obj = (0.8*grid[i_obj][j_obj]) + (0.1*grid[i_p1][j_p1]) + (0.1*grid[i_p2][j_p2])
    return q_obj * 0.9

def update_grid_values(
        grid : np.ndarray
        ) -> tuple:
    m,n = grid.shape
    new_grid = grid.copy()
    for i in range(m):
        for j in range(n):
            dir_list = list()
            if grid[i][j] in [1.0, -1.0, -2.0] :
                new_grid[i][j] = grid[i][j]
                continue
            for dir in range(4):
                dir_list.append(Q_value(grid, i, j, dir))
            max_value = np.max(dir_list)
            new_grid[i][j] = max_value
    
    return new_grid, loss(grid, new_grid)

def get_grid_dirs(
        grid : np.ndarray
        ) -> np.ndarray:
    m,n = grid.shape
    dir_grid = grid.copy().astype(str)
    for i in range(m):
        for j in range(n):
            dir_list = list()
            if grid[i][j] in [1.0, -1.0, -2.0] :
                if grid[i][j] == -2.0:
                    dir_grid[i][j] = "WALL"
                else:
                    dir_grid[i][j] = grid[i][j]
                continue
            for dir in range(4):
                dir_list.append(Q_value(grid, i, j, dir))
            max_dir = np.argmax(dir_list)
            dir_grid[i][j] = dirs[max_dir]
    return dir_grid
            
def loss(
        old_grid : np.ndarray, 
        new_grid : np.ndarray
        ) -> float:
    return np.max(new_grid - old_grid)

def learn(
        grid : np.ndarray,
        iters : int = 100,
        thrishold : float = 1e-9
        ) -> tuple:
    for i in range(iters):
        grid, los = update_grid_values(grid)
        print()
        print("=" *5 + f"iter: {i+1}/{iters}, loss: {los:0.4f}".center(10) + "=" *5 )
        print()
        print_grid(np.round(grid, 2))
        if los <= thrishold:
            break
    print()
    print("=" *10 + "POLICY".center(9) + "=" *10 )
    print_grid(get_grid_dirs(grid))
    


def print_grid(
        grid : np.ndarray
        ):
    m,n = grid.shape
    for i in range(m):
        for j in range(n):
            element = grid[i][j] 
            print(str("WALL" if element == -2.0 else element).center(6), end="|")
        print()


if __name__ == "__main__":
    grid = setup(
        (3,4),
        gyms= [3],
        halls=[7],
        walls=[5]
    )
    print(grid)
    print("=" * 20)

    learn(grid)

          


# 0, 3
# 1, 2
# 0, 2
# 0.72