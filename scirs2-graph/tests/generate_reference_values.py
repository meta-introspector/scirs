#!/usr/bin/env python3
"""
Generate reference values using NetworkX for numerical validation of scirs2-graph.

This script creates test graphs and computes various metrics using NetworkX,
which serves as the reference implementation for validating numerical accuracy.
"""

import networkx as nx
import numpy as np
import json
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt

class ReferenceValueGenerator:
    def __init__(self):
        self.results = {}
    
    def generate_pagerank_reference(self) -> Dict[str, Any]:
        """Generate PageRank reference values for test graphs."""
        print("Generating PageRank reference values...")
        
        # Test case 1: Simple 4-node directed graph
        G1 = nx.DiGraph()
        G1.add_edges_from([
            (0, 1), (0, 2),  # 0 -> 1, 2
            (1, 2),          # 1 -> 2
            (2, 0),          # 2 -> 0
            (3, 0), (3, 1), (3, 2)  # 3 -> 0, 1, 2
        ])
        
        pr1 = nx.pagerank(G1, alpha=0.85, max_iter=100, tol=1e-9)
        
        # Test case 2: Complete graph
        G2 = nx.complete_graph(10)
        pr2 = nx.pagerank(G2, alpha=0.85, max_iter=100, tol=1e-9)
        
        # Test case 3: Path graph
        G3 = nx.path_graph(5)
        pr3 = nx.pagerank(G3, alpha=0.85, max_iter=100, tol=1e-9)
        
        return {
            "simple_directed": {
                "graph": list(G1.edges()),
                "pagerank": pr1,
                "alpha": 0.85
            },
            "complete_10": {
                "n": 10,
                "pagerank": pr2,
                "expected_uniform": 1.0 / 10
            },
            "path_5": {
                "n": 5,
                "pagerank": pr3
            }
        }
    
    def generate_betweenness_reference(self) -> Dict[str, Any]:
        """Generate betweenness centrality reference values."""
        print("Generating betweenness centrality reference values...")
        
        # Path graph
        G1 = nx.path_graph(5)
        bc1 = nx.betweenness_centrality(G1, normalized=False)
        
        # Star graph
        G2 = nx.star_graph(5)  # 6 nodes total (center + 5 leaves)
        bc2 = nx.betweenness_centrality(G2, normalized=False)
        
        # Cycle graph
        G3 = nx.cycle_graph(6)
        bc3 = nx.betweenness_centrality(G3, normalized=False)
        
        return {
            "path_5": {
                "betweenness": bc1,
                "expected_pattern": "Endpoints=0, increases to center"
            },
            "star_6": {
                "betweenness": bc2,
                "center_node": 0,
                "expected_center": (5 * 4) / 2  # (n-1)(n-2)/2
            },
            "cycle_6": {
                "betweenness": bc3,
                "expected": "All nodes equal"
            }
        }
    
    def generate_clustering_reference(self) -> Dict[str, Any]:
        """Generate clustering coefficient reference values."""
        print("Generating clustering coefficient reference values...")
        
        # Complete graph
        G1 = nx.complete_graph(5)
        cc1_global = nx.average_clustering(G1)
        cc1_local = nx.clustering(G1)
        
        # Tree
        G2 = nx.Graph()
        G2.add_edges_from([(0, 1), (0, 2), (1, 3), (1, 4), (2, 5)])
        cc2_global = nx.average_clustering(G2)
        
        # Triangle with tail
        G3 = nx.Graph()
        G3.add_edges_from([(0, 1), (1, 2), (2, 0), (2, 3)])
        cc3_local = nx.clustering(G3)
        cc3_global = nx.average_clustering(G3)
        
        return {
            "complete_5": {
                "global_clustering": cc1_global,
                "local_clustering": cc1_local,
                "expected": 1.0
            },
            "tree": {
                "global_clustering": cc2_global,
                "expected": 0.0
            },
            "triangle_with_tail": {
                "edges": list(G3.edges()),
                "local_clustering": cc3_local,
                "global_clustering": cc3_global,
                "node_2_expected": 1/3
            }
        }
    
    def generate_shortest_path_reference(self) -> Dict[str, Any]:
        """Generate shortest path reference values."""
        print("Generating shortest path reference values...")
        
        # Weighted graph where shortest != fewest edges
        G = nx.Graph()
        G.add_weighted_edges_from([
            (0, 1, 1.0),
            (1, 2, 1.0),
            (2, 4, 1.0),  # Path 0->1->2->4 = 3.0
            (0, 3, 2.0),
            (3, 4, 0.5),  # Path 0->3->4 = 2.5
        ])
        
        path = nx.shortest_path(G, 0, 4, weight='weight')
        length = nx.shortest_path_length(G, 0, 4, weight='weight')
        
        # All pairs shortest paths
        apsp = dict(nx.floyd_warshall(G))
        
        return {
            "weighted_graph": {
                "edges": [(u, v, d['weight']) for u, v, d in G.edges(data=True)],
                "shortest_path_0_4": path,
                "path_length_0_4": length,
                "all_pairs_distances": {
                    f"{i},{j}": apsp[i][j] 
                    for i in range(5) for j in range(5)
                }
            }
        }
    
    def generate_eigenvector_centrality_reference(self) -> Dict[str, Any]:
        """Generate eigenvector centrality reference values."""
        print("Generating eigenvector centrality reference values...")
        
        # Simple connected graph
        G = nx.Graph()
        G.add_edges_from([
            (0, 1), (1, 2), (2, 3), (3, 0), (1, 3)
        ])
        
        ec = nx.eigenvector_centrality(G, max_iter=100, tol=1e-6)
        
        # Verify normalization
        values = list(ec.values())
        norm = np.sqrt(sum(v**2 for v in values))
        
        return {
            "simple_graph": {
                "edges": list(G.edges()),
                "eigenvector_centrality": ec,
                "degrees": dict(G.degree()),
                "l2_norm": norm,
                "expected_properties": [
                    "Nodes 1,3 (degree 3) > nodes 0,2 (degree 2)",
                    "L2 norm = 1.0"
                ]
            }
        }
    
    def generate_spectral_reference(self) -> Dict[str, Any]:
        """Generate spectral graph theory reference values."""
        print("Generating spectral reference values...")
        
        # Cycle graph
        G1 = nx.cycle_graph(4)
        L1 = nx.laplacian_matrix(G1).todense()
        
        # Complete graph
        G2 = nx.complete_graph(5)
        A2 = nx.adjacency_matrix(G2).todense()
        eigenvalues2 = np.linalg.eigvals(A2)
        spectral_radius2 = max(abs(eigenvalues2))
        
        return {
            "cycle_4_laplacian": {
                "laplacian": L1.tolist(),
                "expected_diagonal": 2,
                "expected_pattern": "Circulant matrix"
            },
            "complete_5_spectral": {
                "spectral_radius": float(spectral_radius2.real),
                "expected": 4.0  # n-1 for complete graph
            }
        }
    
    def generate_flow_reference(self) -> Dict[str, Any]:
        """Generate maximum flow reference values."""
        print("Generating maximum flow reference values...")
        
        # Classic flow network
        G = nx.DiGraph()
        capacities = [
            (0, 1, 10), (0, 2, 10),
            (1, 2, 2), (1, 3, 4), (1, 4, 8),
            (2, 4, 9),
            (3, 5, 10),
            (4, 3, 6), (4, 5, 10)
        ]
        
        for u, v, cap in capacities:
            G.add_edge(u, v, capacity=cap)
        
        flow_value, flow_dict = nx.maximum_flow(G, 0, 5)
        
        return {
            "flow_network": {
                "edges": capacities,
                "source": 0,
                "sink": 5,
                "max_flow_value": flow_value,
                "flow_dict": flow_dict
            }
        }
    
    def generate_katz_centrality_reference(self) -> Dict[str, Any]:
        """Generate Katz centrality reference values."""
        print("Generating Katz centrality reference values...")
        
        # Linear graph
        G = nx.path_graph(4)
        alpha = 0.1
        beta = 1.0
        
        kc = nx.katz_centrality(G, alpha=alpha, beta=beta, max_iter=100, tol=1e-6)
        
        return {
            "path_4": {
                "edges": list(G.edges()),
                "alpha": alpha,
                "beta": beta,
                "katz_centrality": kc,
                "expected_pattern": "Central nodes (1,2) > endpoints (0,3)"
            }
        }
    
    def generate_all_references(self) -> Dict[str, Any]:
        """Generate all reference values."""
        return {
            "pagerank": self.generate_pagerank_reference(),
            "betweenness": self.generate_betweenness_reference(),
            "clustering": self.generate_clustering_reference(),
            "shortest_paths": self.generate_shortest_path_reference(),
            "eigenvector_centrality": self.generate_eigenvector_centrality_reference(),
            "spectral": self.generate_spectral_reference(),
            "max_flow": self.generate_flow_reference(),
            "katz_centrality": self.generate_katz_centrality_reference()
        }
    
    def save_references(self, filename: str = "reference_values.json"):
        """Save reference values to JSON file."""
        references = self.generate_all_references()
        
        # Convert numpy types to Python types for JSON serialization
        def convert_types(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(v) for v in obj]
            return obj
        
        references = convert_types(references)
        
        with open(filename, 'w') as f:
            json.dump(references, f, indent=2)
        
        print(f"\nReference values saved to {filename}")
        
        # Also save a human-readable summary
        self.create_summary_report(references)
    
    def create_summary_report(self, references: Dict[str, Any]):
        """Create a human-readable summary of reference values."""
        with open("reference_summary.txt", 'w') as f:
            f.write("NetworkX Reference Values Summary\n")
            f.write("=" * 50 + "\n\n")
            
            # PageRank summary
            f.write("PageRank Values:\n")
            f.write("-" * 20 + "\n")
            pr = references["pagerank"]["simple_directed"]["pagerank"]
            for node, value in pr.items():
                f.write(f"  Node {node}: {value:.6f}\n")
            f.write("\n")
            
            # Betweenness summary
            f.write("Betweenness Centrality (Path Graph):\n")
            f.write("-" * 20 + "\n")
            bc = references["betweenness"]["path_5"]["betweenness"]
            for node, value in bc.items():
                f.write(f"  Node {node}: {value:.1f}\n")
            f.write("\n")
            
            # Clustering summary
            f.write("Clustering Coefficients:\n")
            f.write("-" * 20 + "\n")
            f.write(f"  Complete graph: {references['clustering']['complete_5']['global_clustering']:.3f}\n")
            f.write(f"  Tree: {references['clustering']['tree']['global_clustering']:.3f}\n")
            f.write("\n")
            
            # Shortest path summary
            f.write("Shortest Path (Weighted):\n")
            f.write("-" * 20 + "\n")
            sp = references["shortest_paths"]["weighted_graph"]
            f.write(f"  Path 0->4: {sp['shortest_path_0_4']}\n")
            f.write(f"  Length: {sp['path_length_0_4']}\n")
            f.write("\n")
            
            # Max flow summary
            f.write("Maximum Flow:\n")
            f.write("-" * 20 + "\n")
            f.write(f"  Value: {references['max_flow']['flow_network']['max_flow_value']}\n")
        
        print("Summary report saved to reference_summary.txt")

def visualize_test_graphs():
    """Create visualizations of test graphs for documentation."""
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()
    
    # Graph 1: Simple directed for PageRank
    G1 = nx.DiGraph()
    G1.add_edges_from([(0, 1), (0, 2), (1, 2), (2, 0), (3, 0), (3, 1), (3, 2)])
    nx.draw(G1, ax=axes[0], with_labels=True, node_color='lightblue', 
            arrows=True, arrowsize=20)
    axes[0].set_title("PageRank Test Graph")
    
    # Graph 2: Path graph
    G2 = nx.path_graph(5)
    nx.draw(G2, ax=axes[1], with_labels=True, node_color='lightgreen',
            pos=nx.spring_layout(G2))
    axes[1].set_title("Path Graph (n=5)")
    
    # Graph 3: Star graph
    G3 = nx.star_graph(5)
    nx.draw(G3, ax=axes[2], with_labels=True, node_color='lightcoral',
            pos=nx.spring_layout(G3))
    axes[2].set_title("Star Graph (n=6)")
    
    # Graph 4: Triangle with tail
    G4 = nx.Graph()
    G4.add_edges_from([(0, 1), (1, 2), (2, 0), (2, 3)])
    nx.draw(G4, ax=axes[3], with_labels=True, node_color='lightyellow',
            pos=nx.spring_layout(G4))
    axes[3].set_title("Triangle with Tail")
    
    # Graph 5: Weighted shortest path
    G5 = nx.Graph()
    G5.add_weighted_edges_from([(0, 1, 1), (1, 2, 1), (2, 4, 1), 
                                (0, 3, 2), (3, 4, 0.5)])
    pos = nx.spring_layout(G5)
    nx.draw(G5, ax=axes[4], pos=pos, with_labels=True, node_color='lightpink')
    labels = nx.get_edge_attributes(G5, 'weight')
    nx.draw_networkx_edge_labels(G5, pos, labels, ax=axes[4])
    axes[4].set_title("Weighted Graph")
    
    # Graph 6: Flow network
    G6 = nx.DiGraph()
    G6.add_weighted_edges_from([
        (0, 1, 10), (0, 2, 10),
        (1, 2, 2), (1, 3, 4), (1, 4, 8),
        (2, 4, 9),
        (3, 5, 10),
        (4, 3, 6), (4, 5, 10)
    ])
    pos = nx.spring_layout(G6)
    nx.draw(G6, ax=axes[5], pos=pos, with_labels=True, node_color='lavender',
            arrows=True, arrowsize=20)
    labels = nx.get_edge_attributes(G6, 'weight')
    nx.draw_networkx_edge_labels(G6, pos, labels, ax=axes[5])
    axes[5].set_title("Flow Network")
    
    plt.tight_layout()
    plt.savefig("test_graphs_visualization.png", dpi=150)
    print("Test graphs visualization saved to test_graphs_visualization.png")

def main():
    print("Generating reference values for scirs2-graph numerical validation")
    print("=" * 60)
    print(f"NetworkX version: {nx.__version__}")
    print()
    
    generator = ReferenceValueGenerator()
    generator.save_references()
    
    print("\nCreating test graph visualizations...")
    visualize_test_graphs()
    
    print("\nDone! Files created:")
    print("  - reference_values.json: Complete reference data")
    print("  - reference_summary.txt: Human-readable summary")
    print("  - test_graphs_visualization.png: Visual representation of test graphs")

if __name__ == "__main__":
    main()