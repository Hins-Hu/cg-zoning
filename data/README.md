# Data Directory

The experiment scripts read all inputs from this directory, running from the
repo root. Data files are not shipped in the repository (the five paper
cities total about 810 MB). Only this README and `cities.ipynb` are tracked
by git. You can run the code on any city by providing files in the formats
below.

## File formats

For a city with short name `{city}` (the paper uses `miami`, `boston`,
`atlanta`, `chatt`, `nashville`):

| File | Format |
| --- | --- |
| `demand_{city}.csv` | One row per trip, columns `origin_lat`, `origin_lon`, `destination_lat`, `destination_lon` (EPSG:4269 coordinates) |
| `{city}_graph.graphml` | Road network from OpenStreetMap, saved by `osmnx` (`ox.graph_from_place`, then `ox.save_graphml`) |
| `center_distances_{city}_res{r}.json` | Nested dict keyed by hex id: shortest-path travel time between the road-network nodes nearest to each pair of H3 cell centroids, for each resolution `r` you want to run |
| `hex_{city}_res{r}.geojson` | H3 hex map of the city at resolution `r` (only needed by the equity pipeline) |

To run your own city: pick a short name, place a demand CSV and a road
network under these names, and pre-compute the center-distance JSON.
`cities.ipynb` (in this directory) is the reference implementation of these
generation steps, including the hex maps and the distance caches. The
distance computation is all-pairs shortest paths and can take a while,
which is why the experiment scripts load the cached JSON instead of
recomputing it.

## Where the paper's demand data comes from

- **Miami, Boston, Atlanta, Nashville:** synthetic OD trips generated with
  [MoveOD](https://github.com/rishavsen1/move_od), which synthesizes
  origin-destination demand from public U.S. Census sources (LODES commute
  flows and building footprints) and outputs trips in exactly this schema.
  See the MoveOD repository and its paper
  ([arXiv:2510.18858](https://arxiv.org/abs/2510.18858)) for the
  methodology. You can regenerate demand for these or any other U.S. city
  with MoveOD directly.
- **Chattanooga:** a weekday OD extract from a proprietary commercial
  mobility dataset. Its license permits neither redistribution of the raw
  data nor disclosure of the vendor, so that CSV cannot be shipped. Any OD
  source in the same schema (for example a MoveOD-generated table) can be
  placed at `demand_chatt.csv` instead.

The experiment scripts make no distinction between data sources. Whatever
`demand_{city}.csv` contains is what gets aggregated and served.
