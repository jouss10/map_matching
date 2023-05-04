from typing import List

import geopandas as gpd
import networkx as nx
import osmnx as ox
from config import USED_CRS
from streamlit import cache_data


@cache_data()
def load_gps_points(file_path: str) -> gpd.GeoDataFrame:
    points = gpd.read_file(file_path)
    return points.to_crs(USED_CRS)


def load_osmnx_graph(points: gpd.GeoDataFrame) -> nx.MultiDiGraph:
    bbox = _get_bbox_from_points(points)
    return ox.graph_from_bbox(
        *bbox,
        network_type="drive",
        simplify=True,
        retain_all=False,
    )


def _get_bbox_from_points(df: gpd.GeoDataFrame, step: float = 0.01) -> List[float]:
    df = df.assign(
        longitude=df["geometry"].apply(lambda p: p.x),
        latitude=df["geometry"].apply(lambda p: p.y),
    )
    lon_min, lat_min = df["longitude"].min(), df["latitude"].min()
    lon_max, lat_max = df["longitude"].max(), df["latitude"].max()
    return [
        lat_max + step,
        lat_min - step,
        lon_max + step,
        lon_min - step,
    ]
