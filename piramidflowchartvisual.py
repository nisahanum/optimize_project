import matplotlib.pyplot as plt
import networkx as nx

# Function to create and plot a hierarchical flowchart
def plot_flowchart():
    G = nx.DiGraph()

    # Define nodes for Flowchart structure
    nodes = {
        "Project Interdependencies": (3, 4),
        "Project Interdependency Management": (3, 3),
        "Hard Practices": (2, 2),
        "Soft Practices": (4, 2),
        "Portfolio Success": (2, 1),
        "Portfolio Failure": (4, 1)
    }

    # Add nodes to graph
    for node, pos in nodes.items():
        G.add_node(node, pos=pos)

    # Define edges
    edges = [
        ("Project Interdependencies", "Project Interdependency Management"),
        ("Project Interdependency Management", "Hard Practices"),
        ("Project Interdependency Management", "Soft Practices"),
        ("Hard Practices", "Portfolio Success"),
        ("Soft Practices", "Portfolio Success"),
        ("Hard Practices", "Portfolio Failure"),
        ("Soft Practices", "Portfolio Failure")
    ]

    G.add_edges_from(edges)

    # Get positions
    pos = {node: (x, y) for node, (x, y) in nodes.items()}

    # Draw flowchart
    plt.figure(figsize=(8, 6))
    nx.draw(G, pos, with_labels=True, node_color="lightblue", edge_color="gray", node_size=3000, font_size=9, font_weight="bold")
    plt.title("Flowchart Model of Project Interdependency Management")
    plt.show()

# Function to create and plot a Pyramid Model
def plot_pyramid():
    G = nx.DiGraph()

    # Define nodes for Pyramid structure
    nodes = {
        "Project Interdependencies": (3, 3),
        "Project Interdependency Management": (3, 2),
        "Portfolio Success": (2, 1),
        "Portfolio Failure": (4, 1),
        "Hard Practices": (2, 2),
        "Soft Practices": (4, 2)
    }

    # Add nodes to graph
    for node, pos in nodes.items():
        G.add_node(node, pos=pos)

    # Define edges for hierarchy (Pyramid structure)
    edges = [
        ("Project Interdependencies", "Project Interdependency Management"),
        ("Project Interdependency Management", "Hard Practices"),
        ("Project Interdependency Management", "Soft Practices"),
        ("Hard Practices", "Portfolio Success"),
        ("Soft Practices", "Portfolio Success"),
        ("Hard Practices", "Portfolio Failure"),
        ("Soft Practices", "Portfolio Failure")
    ]

    G.add_edges_from(edges)

    # Get positions for nodes
    pos = {node: (x, y) for node, (x, y) in nodes.items()}

    # Draw the pyramid model
    plt.figure(figsize=(8, 6))
    nx.draw(G, pos, with_labels=True, node_color="lightgreen", edge_color="gray", node_size=3000, font_size=9, font_weight="bold")
    plt.title("Pyramid Model of Project Interdependency Management")
    plt.show()

# Plot both models
plot_flowchart()
plot_pyramid()
