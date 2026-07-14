"""Cost-preserving zone patching (Algorithm 2 in the paper appendix).

repair_zones absorbs into each selected zone every candidate cell whose
maximum distance to the zone does not exceed the zone's current diameter, so
connectivity holes are filled without increasing the zone cost. Used by the
CG entry point's --post_process_connectivity step (Figure 4).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ZoneRepairStats:
    zone_index: int
    original_size: int
    repaired_size: int
    added_cells: list[str]


def repair_zones(
    zones: list[set[str] | frozenset[str]],
    distances: dict[str, dict[str, float]],
    candidate_cells: set[str] | list[str] | tuple[str, ...] | None = None,
) -> tuple[list[set[str]], list[ZoneRepairStats]]:
    """Absorb every cost-free cell into each selected zone.

    The repair rule follows the intended MTZ connectivity-repair version:
    iterate over the full candidate set V and absorb any cell g whose maximum
    distance to the current zone S does not exceed the current zone diameter.
    Sequential absorption is important: the zone is updated immediately after
    each accepted cell so mutually incompatible candidates are filtered out.
    """

    if candidate_cells is None:
        candidate_cells = sorted(distances.keys())
    else:
        candidate_cells = sorted(set(candidate_cells))

    repaired_zones: list[set[str]] = []
    stats: list[ZoneRepairStats] = []

    for idx, zone in enumerate(zones):
        original_zone = set(zone)
        repaired_zone = set(zone)
        zone_diameter = _zone_diameter(repaired_zone, distances)
        tol = 1e-7 * max(zone_diameter, 1.0)

        while True:
            progress = False
            for cell in candidate_cells:
                if cell in repaired_zone:
                    continue
                if _max_distance_to_zone(cell, repaired_zone, distances) <= zone_diameter + tol:
                    repaired_zone.add(cell)
                    progress = True
            if not progress:
                break

        added_cells = sorted(repaired_zone - original_zone)
        repaired_zones.append(repaired_zone)
        stats.append(
            ZoneRepairStats(
                zone_index=idx,
                original_size=len(original_zone),
                repaired_size=len(repaired_zone),
                added_cells=added_cells,
            )
        )

    return repaired_zones, stats


def _zone_diameter(zone: set[str], distances: dict[str, dict[str, float]]) -> float:
    """Return the maximum pairwise distance inside the zone."""

    if len(zone) <= 1:
        return 0.0

    zone_list = list(zone)
    max_distance = 0.0
    for i, source in enumerate(zone_list):
        for target in zone_list[i + 1:]:
            max_distance = max(max_distance, float(distances[source][target]))
    return max_distance


def _max_distance_to_zone(cell: str, zone: set[str], distances: dict[str, dict[str, float]]) -> float:
    """Return max_j max{c(cell, j), c(j, cell)} over the current zone."""

    max_distance = 0.0
    for zone_cell in zone:
        max_distance = max(
            max_distance,
            float(distances[cell][zone_cell]),
            float(distances[zone_cell][cell]),
        )
    return max_distance
