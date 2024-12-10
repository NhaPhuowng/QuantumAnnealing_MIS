import numpy as np

def calculate_average_and_std(arr):
    # Calculate the average (mean) and standard deviation (std)
    average = np.mean(arr)
    std_deviation = np.std(arr)
    
    # Print the results
    print(f"Average: {average}")
    print(f"Standard Deviation: {std_deviation}")

# Example usage:
array = [156, 312, 468, 624]
calculate_average_and_std(array)
