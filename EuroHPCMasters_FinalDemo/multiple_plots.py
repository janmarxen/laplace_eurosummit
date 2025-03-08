import os
import struct
import numpy as np
import matplotlib.pyplot as plt

def create_heatmap(data, ax):
    # Reshape the data into a list of (x, y, value) tuples
    points = [(data[i], data[i+1], data[i+2]) for i in range(0, len(data), 3)]
    
    # Extract x, y, and values
    x = [point[0] for point in points]
    y = [point[1] for point in points]
    values = [point[2] for point in points]
    
    # Create a grid for the heatmap
    x_unique = sorted(set(x))
    y_unique = sorted(set(y))
    grid = np.zeros((len(y_unique), len(x_unique)))
    
    # Fill the grid with values
    for point in points:
        x_idx = x_unique.index(point[0])
        y_idx = y_unique.index(point[1])
        grid[y_idx, x_idx] = point[2]
    
    # Plot the heatmap on the given axis
    im = ax.imshow(grid, cmap='jet', interpolation='nearest', origin='lower',
                   extent=[min(x_unique), max(x_unique), min(y_unique), max(y_unique)])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    # ax.set_title('Heat Map')
    return im

# Find all .dat files in the current directory
dat_files = [f for f in os.listdir('.') if f.endswith('.dat')]
# Sort files by modification time (oldest first, like `ls -lrt`)
dat_files.sort(key=lambda x: os.path.getmtime(x))

# Determine the layout of the subplots
num_files = len(dat_files)
rows = int(np.ceil(np.sqrt(num_files)))  # Adjust rows and columns as needed
cols = int(np.ceil(num_files / rows))

# Create a figure to hold all the subplots
fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))  # Adjust figsize as needed
# fig.suptitle('Heatmaps of .dat Files', fontsize=7)

# Flatten the axes array for easy iteration
axes = axes.flatten()

# Loop through each .dat file and plot its heatmap
for i, filename in enumerate(dat_files):
    with open(filename, 'rb') as file:
        binary_data = file.read()
    
    Byte_Order = '<'            # little-endian
    Format_Characters = 'f'     # float (4 bytes)
    data_format = Byte_Order + Format_Characters
    
    r = np.array(list(struct.iter_unpack(data_format, binary_data)), dtype=float)
    single_list = r.flatten().tolist()
    
    # Plot the heatmap in the corresponding subplot
    im = create_heatmap(single_list, axes[i])
    fig.colorbar(im, ax=axes[i], label='Value')

# Hide any unused subplots
for j in range(i + 1, len(axes)):
    axes[j].axis('off')

# Adjust layout and display the plot
# plt.tight_layout()
plt.savefig('solution.png')
plt.savefig('solution.pdf')
# plt.show()