# =============================================================
# scripts/run_defect_stats.py -- entry point (was the __main__ block of graph/defects.py)
# =============================================================

from config import DEFECT_K
from graph.defects import (
    build_defect_subgraph,
    classify_defects_statistical,
    compute_degree_deficit,
    compute_degrees,
    track_defects_over_time,
)


if __name__ == "__main__":
    from data_io.trajectory import load_trajectory
    from graph.build import atoms_to_pyg
    from config import TRAJECTORY_FILE

    trajectory = load_trajectory(TRAJECTORY_FILE)
    graphs     = [atoms_to_pyg(atoms) for atoms in trajectory[:20]]

    g       = graphs[0]
    degrees = compute_degrees(g)
    Z       = g.x.view(-1)
    deficit = compute_degree_deficit(degrees, Z)
    defects = classify_defects_statistical(degrees, Z, k=DEFECT_K)

    print(f"Defects in first frame: {defects.sum().item()}")

    subgraph, _ = build_defect_subgraph(g, defects)
    print("Defect subgraph:", subgraph)

    tracks = track_defects_over_time(graphs)
    print(f"Tracked defects: {len(tracks)}")
    avg_life = sum(len(t) for t in tracks.values()) / len(tracks) if tracks else 0
    print(f"Average defect lifetime: {avg_life:.1f} frames")
