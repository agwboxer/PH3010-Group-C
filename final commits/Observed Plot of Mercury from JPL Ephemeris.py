"""
Created on Sat Nov 16 17:27:09 2024

@author: H Rogers

Extracted the ephemeris and plotted the x position and the x velocity coordinates
From 25 Nov 23' to 21 Feb 24' (88 days) and plotted in AU/day
"""

import matplotlib.pyplot as plt

# Example file_path - Replace with your actual file path
file_path = 'horizons_results.txt'

# Initialize lists to hold the position (x) and velocity (vx) values
x_values = []
vx_values = []

# Open the file and parse the data
with open(file_path, 'r') as file:
    for line in file:
        # Skip lines that are empty
        if line.strip() == '':
            continue
        
        # Split the line by commas (since the data is comma-separated)
        line_parts = line.split(',')
        
        # Ensure the line has at least 6 values (to safely extract x and vx)
        if len(line_parts) >= 6:
            try:
                # Extract x (3rd element) and vx (6th element)
                x = float(line_parts[2])  # 3rd value in the line
                vx = float(line_parts[5])  # 6th value in the line
                
                # Append the extracted values to the lists
                x_values.append(x)
                vx_values.append(vx)
            except ValueError:
                # Skip lines with invalid or missing data
                continue

# Check if valid data was collected
if len(x_values) == 0 or len(vx_values) == 0:
    print("Error: No valid data found to plot.")
else:
    # Plot X vs X velocity
    plt.figure(figsize=(10, 6))
    plt.plot(x_values, vx_values, label='X Position vs X Velocity', color='r', marker='o', linestyle='-')

    # Labels and title
    plt.title('X Position vs X Velocity')
    plt.xlabel('X Position (AU)')
    plt.ylabel('X Velocity (AU/day)')

    # Display the legend
    plt.legend()

    # Show the grid
    plt.grid(True)

    # Show the plot
    plt.show()
