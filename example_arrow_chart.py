import matplotlib.pyplot as plt

# Data for the phases
phases = ["Phase I", "Phase II", "Phase III", "Phase IV", "Commercial"]
y_inhouse = [5, 4, 3, 2, 1]  # Y-positions for in-house arrows
y_outsourcing = [4.5, 3.5, 2.5, 1.5, 0.5]  # Y-positions for outsourcing arrows

# Create the figure
plt.figure(figsize=(12, 6))

# Draw arrows for in-house option
for i, phase in enumerate(phases):
    plt.arrow(i + 1, y_inhouse[i], 0, -0.5, head_width=0.1, head_length=0.2, fc="blue", ec="blue")  # Cash outflows
    plt.arrow(i + 1, y_inhouse[i] - 0.7, 0, 0.5, head_width=0.1, head_length=0.2, fc="green", ec="green")  # Cash inflows

# Draw arrows for outsourcing option
for i, phase in enumerate(phases):
    if phase == "Phase III":  # Example outsourced phase
        plt.text(i + 1, y_outsourcing[i], "X", fontsize=14, color="red", ha="center")  # Red cross for outsourcing
    else:
        plt.arrow(i + 1, y_outsourcing[i], 0, -0.5, head_width=0.1, head_length=0.2, fc="blue", ec="blue")  # Cash outflows
        plt.arrow(i + 1, y_outsourcing[i] - 0.7, 0, 0.5, head_width=0.1, head_length=0.2, fc="green", ec="green")  # Cash inflows

# Add phase labels
for i, phase in enumerate(phases):
    plt.text(i + 1, 5.5, phase, fontsize=12, ha="center", color="black")

# Styling
plt.title("Timeline-Based Cash Flow Diagram: In-house vs. Outsourcing", fontsize=14)
plt.xlabel("Project Phases")
plt.ylabel("Cash Flow Direction")
plt.xticks(range(1, len(phases) + 1), phases)
plt.yticks([])
plt.axhline(y=0, color="black", linewidth=0.8)  # Horizontal axis
plt.legend(["In-house", "Outsourcing"], loc="upper right")
plt.tight_layout()

# Show plot
plt.show()
