import matplotlib.pyplot as plt

# Selected projects and their costs
selected_projects = ['Project A', 'Project C']
selected_costs = [112000, 86300]

# Create a pie chart
plt.figure(figsize=(7, 7))
plt.pie(selected_costs, labels=selected_projects, autopct='%1.1f%%', startangle=140, colors=['lightblue', 'lightgreen'])
plt.title('Cost Distribution of Selected Projects')
plt.axis('equal')  # Equal aspect ratio ensures that pie chart is a circle.
plt.show()