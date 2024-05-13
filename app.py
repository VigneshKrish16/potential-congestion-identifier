import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from math import radians, sin, cos, sqrt, atan2
import plotly.graph_objects as go
from geopy.distance import geodesic

def generate_visualizations(data_preprocessed):
    flight_paths = data_preprocessed.groupby('flight_name')

    # 3D Visualization
    fig_3d = go.Figure()

    for flight, path in flight_paths:
        lats = path['latitude']
        lons = path['longitude']
        alts = path['altitude']

        flight_trace_3d = go.Scatter3d(
            x=lons,
            y=lats,
            z=alts,
            mode='lines+markers',
            name=flight,
            marker=dict(
                size=5,
                color=np.random.rand(3,)
            ),
            hovertemplate='<b>Flight: %{text}</b><br>' +
                           'Latitude: %{y:.5f}<br>' +
                           'Longitude: %{x:.5f}<br>' +
                           'Altitude: %{z:.0f} m<extra></extra>',
            text=[f'Waypoint {i+1}' for i in range(len(lats))]
        )
        fig_3d.add_trace(flight_trace_3d)

    # Add congestion area markers
    kmeans = KMeans(n_clusters=2).fit(data_preprocessed[['latitude', 'longitude', 'altitude']])
    cluster_centers = kmeans.cluster_centers_
    cluster_counts = np.bincount(kmeans.labels_)

    congestion_areas = []
    for i in range(len(cluster_centers)):
        if cluster_counts[i] > np.percentile(cluster_counts, 25):
            congestion_areas.append(cluster_centers[i])

    for area in congestion_areas:
        fig_3d.add_trace(go.Scatter3d(
            x=[area[1]],
            y=[area[0]],
            z=[area[2]],
            mode='markers',
            marker=dict(
                size=10,
                color='red'
            ),
            name='Congestion Area'
        ))

    fig_3d.update_layout(
        scene=dict(
            xaxis_title='Longitude',
            yaxis_title='Latitude',
            zaxis_title='Altitude (m)',
            camera=dict(
                eye=dict(x=0, y=0, z=2.5)
            )
        ),
        title='3D Visualization of Flight Paths with Congestion Areas',
        legend=dict(
            x=0.1,
            y=1,
            traceorder='normal',
            font=dict(
                family='sans-serif',
                size=12,
                color='black'
            ),
            bgcolor='LightSteelBlue',
            bordercolor='Black',
            borderwidth=2
        ),
        updatemenus=[
            dict(
                type="buttons",
                buttons=[
                    dict(
                        label="Play",
                        method="animate",
                        args=[None]
                    )
                ]
            )
        ]
    )

    # Line Plot
    fig_line = go.Figure()

    for flight, path in flight_paths:
        lats = path['latitude']
        lons = path['longitude']
        alts = path['altitude']

        fig_line.add_trace(go.Scatter(
            x=lons,
            y=alts,
            mode='lines+markers',
            name=flight,
            marker=dict(
                color=np.random.rand(3,)
            )
        ))

    fig_line.update_layout(
        title='Altitude Profile of Flight Paths',
        xaxis_title='Longitude',
        yaxis_title='Altitude (m)',
        legend=dict(
            x=0.1,
            y=1,
            traceorder='normal',
            font=dict(
                family='sans-serif',
                size=12,
                color='black'
            ),
            bgcolor='LightSteelBlue',
            bordercolor='Black',
            borderwidth=2
        )
    )

    st.plotly_chart(fig_3d, use_container_width=True)
    st.plotly_chart(fig_line, use_container_width=True)


def main():
    st.title("PCI Project: UAV Flight Analysis")
    st.write("Welcome to the UAV Flight Analysis App!")

    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        data = data.dropna()

        st.write("Uploaded Dataset:")
        st.write(data)

        imputer = SimpleImputer(strategy='mean')
        data_preprocessed = imputer.fit_transform(data[['latitude', 'longitude', 'altitude']])

        # Create a new DataFrame with the preprocessed data
        data_preprocessed = pd.DataFrame(data_preprocessed, columns=['latitude', 'longitude', 'altitude'])
        data_preprocessed['flight_name'] = data['flight_name']  # Add the 'flight_name' column

        kmeans = KMeans(n_clusters=2).fit(data_preprocessed[['latitude', 'longitude', 'altitude']])
        cluster_centers = kmeans.cluster_centers_
        cluster_counts = np.bincount(kmeans.labels_)

        congestion_areas = []
        for i in range(len(cluster_centers)):
            if cluster_counts[i] > np.percentile(cluster_counts, 25):
                congestion_areas.append(cluster_centers[i])

        st.subheader("Potential Congestion Areas:")
        for i, area in enumerate(congestion_areas, start=1):
            st.write(f"Congestion Area {i}:")
            st.write(f"Latitude: {area[0]}, Longitude: {area[1]}, Altitude: {area[2]}")

        conflicts = []
        for i, row1 in data.iterrows():
            for j, row2 in data.iterrows():
                if i != j:
                    d = geodesic((row1['latitude'], row1['longitude']), (row2['latitude'], row2['longitude'])).km
                    if d < 1:
                        conflicts.append((row1, row2))

        st.subheader("Potential Conflicts:")
        for conflict in conflicts:
            st.write(f"Conflict between UAVs at {conflict[0]} and {conflict[1]}")

        generate_visualizations(data_preprocessed)

if __name__ == "__main__":
    main()