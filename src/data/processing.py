import uuid
from typing import List

import geopandas as gpd
import pandas as pd
from config import PROJECTION_CRS, USED_CRS


def get_road_segment_candidates(
    points: gpd.GeoDataFrame,
    edges: gpd.GeoDataFrame,
    max_search_radius: int,
    nb_nearest_neighbors: int,
) -> gpd.GeoDataFrame:
    points["geometry"] = (
        points["point_geometry"]
        .set_crs(USED_CRS)
        .to_crs(PROJECTION_CRS)
        .buffer(distance=max_search_radius)
        .to_crs(USED_CRS)
    )
    result = gpd.sjoin_nearest(points, edges.reset_index(), how="inner")
    result["distance"] = gpd.GeoSeries(result["point_geometry"]).distance(
        gpd.GeoSeries(result["edges_geometry"])
    )
    result = result.sort_values(by="distance", ascending=True)
    candidates = result.groupby(["trip_id", "unixtime"]).head(nb_nearest_neighbors)

    candidates["projection_point"] = candidates.apply(
        lambda row: row["edges_geometry"].interpolate(
            row["edges_geometry"].project(row[["point_geometry"]])
        ),
        axis=1,
    )
    return candidates


def sample_points(
    points: gpd.GeoDataFrame, sampling_frequency: int
) -> gpd.GeoDataFrame:
    points["unixtime"] = pd.to_datetime(points["unixtime"], unit="s")

    sampled_points = (
        points.set_index("unixtime")
        .groupby("trip_id")
        .resample(f"{sampling_frequency}s")
        .last()
        .reset_index(level="unixtime")
        .reset_index(drop=True)
        .dropna()
    )

    return sampled_points


def create_uuid_from_cols(
    df: gpd.GeoDataFrame, uuid_column: str, cols: List[str]
) -> gpd.GeoDataFrame:
    return df.assign(
        **{
            uuid_column: df.apply(
                lambda row: str(
                    uuid.uuid3(
                        uuid.NAMESPACE_DNS,
                        "".join([str(row[col]) for col in cols]),
                    )
                ),
                axis=1,
            )
        }
    )
