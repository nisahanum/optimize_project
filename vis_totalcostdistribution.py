import matplotlib.pyplot as plt

# Project names and total costs
project_names = ['Project A', 'Project B', 'Project C']
total_costs = [112000, 168000, 86300]

# Create a bar chart
plt.figure(figsize=(8, 5))
plt.bar(project_names, total_costs, color=['blue', 'orange', 'green'])
plt.title('Total Costs of Each Project')
plt.xlabel('Projects')
plt.ylabel('Total Cost ($)')
plt.grid(axis='y')
plt.xticks(rotation=45)
plt.show()