import geopandas as gpd
import pandas as pd
import pydeck as pdk
import streamlit as st


def plot_trip_matching(result_df: gpd.GeoDataFrame) -> None:
    scatter_df = result_df.dropna(subset="point_geometry")
    scatter_df = scatter_df.assign(
        lon=scatter_df["point_geometry"].apply(lambda p: round(p.x, 5)),
        lat=scatter_df["point_geometry"].apply(lambda p: round(p.y, 5)),
    )

    path_df = result_df.assign(
        path=result_df["edges_geometry"].apply(lambda x: list(x.coords)),
        datetime=pd.to_datetime(result_df["unixtime"]),
    )
    st.pydeck_chart(
        pdk.Deck(
            map_style=None,
            initial_view_state=pdk.ViewState(
                latitude=52.38,
                longitude=9.71,
                zoom=11,
                pitch=50,
            ),
            layers=[
                pdk.Layer(
                    "ScatterplotLayer",
                    data=scatter_df[["lon", "lat", "observation"]],
                    get_position="[lon, lat]",
                    get_radius=20,
                    get_color="[244, 243, 10]",
                    pickable=True,
                ),
                pdk.Layer(
                    "PathLayer",
                    data=path_df[["path", "datetime"]],
                    get_path="path",
                    get_color=[255, 165, 0],
                    get_time="datetime",
                    get_width=5,
                    auto_highlight=True,
                    highlight_color=[255, 255, 255],
                    pickable=True,
                ),
            ],
        )
    )
