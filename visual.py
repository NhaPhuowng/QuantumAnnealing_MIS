import matplotlib.pyplot as plt

# Example data: average values and standard deviations
x = [0.5, 1, 2, 3]  # x-axis values (could represent time, categories, etc.)
y = [0, 97.58, 90.1, 90.64]  # average values (y-axis)
std_dev = [0, 5.19, 12.23, 117.78]  # standard deviations (error bars)

# Create the plot with error bars
plt.errorbar(x, y, yerr=std_dev, fmt='-o', capsize=5, linestyle='-', color='b', ecolor='r', elinewidth=2, capthick=2)

# Add labels and title
plt.xlabel('X-axis')
plt.ylabel('Average Values')
plt.title('Line Chart with Error Bars')

# Show the plot
plt.show()
