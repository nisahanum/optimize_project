import matplotlib.pyplot as plt

# Risk adjustments and AHP scores
risk_adjustments = [0.10, 0.15, 0.05]
AHP_scores = [0.5, 0.3, 0.2]
project_names = ['Project A', 'Project B', 'Project C']

# Create a scatter plot
plt.figure(figsize=(8, 5))
plt.scatter(risk_adjustments, AHP_scores, color='purple')
for i, project in enumerate(project_names):
    plt.annotate(project, (risk_adjustments[i], AHP_scores[i]), textcoords="offset points", xytext=(0,10), ha='center')
plt.title('Risk Adjustment vs. AHP Score')
plt.xlabel('Risk Adjustment')
plt.ylabel('AHP Score')
plt.grid()
plt.show()