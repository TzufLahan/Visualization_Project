import json
import requests
import streamlit as st
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from shapely.geometry import Polygon


def read_csv_from_googledrive(url):
    url = 'https://drive.google.com/uc?id=' + url.split('/')[-2]
    df = pd.read_csv(url)
    return df

# Load the preprocessed DataFrame
preprocessed_flying_etiquette_df = read_csv_from_googledrive('https://drive.google.com/file/d/1H0tBXnR_tYa0pvzVi1XI5mESRDr7Q96j/view?usp=sharing')

# Define available options for filters
income_levels = ['$0 - $24,999', '$25,000 - $49,999', '$50,000 - $99,999', '$100,000 - $149,999', '150000']
genders = ['Male', 'Female']
Region_list = ['Pacific', 'Mountain', 'West North Central', 'West South Central', 'East North Central', 'East South Central', 'South Atlantic', 'Middle Atlantic', 'New England']


def download_geojson(url, local_path):
    response = requests.get(url)
    with open(local_path, 'wb') as file:
        file.write(response.content)

geojson_url = 'https://raw.githubusercontent.com/tzuflahan/Streamlit/main/aggregated_regions.geojson'
local_geojson_path = 'aggregated_regions.geojson'
download_geojson(geojson_url, local_geojson_path)

# Load the shapefile for the US regions
gdf = gpd.read_file(local_geojson_path)

# Ensure all regions are included in the GeoDataFrame
missing_regions = [region for region in Region_list if region not in gdf['region'].values]
for region in missing_regions:
    if region == 'Pacific':
        polygon = Polygon([(-125, 32), (-115, 32), (-115, 42), (-125, 42), (-125, 32)])
    elif region == 'Mountain':
        # Define the coordinates for the Mountain region here
        polygon = Polygon([...])
    # Add other regions as needed
    new_region = gpd.GeoDataFrame({'region': [region], 'geometry': [polygon]})
    gdf = gdf.append(new_region, ignore_index=True)

# Calculate the global min and max for the normalized politeness score
politeness_score_by_filters = []
for selected_income in income_levels:
    for selected_gender in genders:
        for selected_region in Region_list:
            filtered_data = preprocessed_flying_etiquette_df[
                (preprocessed_flying_etiquette_df['Household Income'] == selected_income) &
                (preprocessed_flying_etiquette_df['Gender'] == selected_gender) &
                (preprocessed_flying_etiquette_df['Location'] == selected_region)
            ]
            politeness_score_by_filters.append(filtered_data['politeness_score_normalized'].mean())

global_min = min(politeness_score_by_filters)
global_max = max(politeness_score_by_filters)

# Initialize session state
if 'selected_region' not in st.session_state:
    st.session_state.selected_region = 'Pacific'  # Default region

# Streamlit layout enhancements
st.set_page_config(
    page_title="US Heat Map Politeness Level By Regions",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Title and description
st.title('US Heat Map Politeness Level By Regions')

st.image("https://raw.githubusercontent.com/tzuflahan/Streamlit/main/Flight.png", use_column_width = True, width=50)
st.markdown("""
    <div style='text-align: center'>
        Welcome to the US Regions Politeness Heat Map! This interactive app allows you to explore the average politeness scores 
        across different regions in the United States based on household income and gender.
        <br><br>
        Data was collected from people in different areas across the United States, these people were asked about different situations and interactions between strangers on flights and asked their opinion on the subject 
        We used these findings to build a politeness level for each person and below are our results
    </div>
""", unsafe_allow_html=True)


# Add map description
st.markdown("""
    <div style='text-align: center'>
        <br><br>
        <strong>Please select Household Income and Gender to see the politeness differences between various regions in the United States.</strong>
    </div>
""", unsafe_allow_html=True)

# Sidebar for filters
st.sidebar.header("Filter Options")
selected_income = st.sidebar.selectbox('Select Household Income', income_levels)
selected_gender = st.sidebar.selectbox('Select Gender', genders)

# Filter data based on selections
filtered_data = preprocessed_flying_etiquette_df[
    (preprocessed_flying_etiquette_df['Household Income'] == selected_income) &
    (preprocessed_flying_etiquette_df['Gender'] == selected_gender)
]

# Check if the filtered data is not empty
if not filtered_data.empty:
    # Group by region and calculate the average normalized politeness score
    region_politeness = filtered_data.groupby('Mapped Location')['politeness_score_normalized'].mean().reset_index()
    region_politeness.columns = ['region', 'avg_politeness_score_normalized']

    # Merge the data with the shapefile
    merged = gdf.set_index('region').join(region_politeness.set_index('region'))

    # Convert GeoDataFrame to GeoJSON
    merged_geojson = json.loads(merged.to_json())

    # Create an interactive map with Plotly
    fig = px.choropleth_mapbox(
        merged,
        geojson=merged_geojson,
        locations=merged.index,
        color='avg_politeness_score_normalized',
        color_continuous_scale="Spectral",
        range_color=[global_min, global_max],  # Set the range color to global min and max
        mapbox_style="open-street-map",
        zoom=3,
        center={"lat": 37.0902, "lon": -95.7129},
        opacity=0.5,
        labels={'avg_politeness_score_normalized': 'Avg Politeness Score'}
    )

    # Calculate centroids for each region to place the text annotations
    gdf['centroid'] = gdf.centroid
    for idx, row in gdf.iterrows():
        fig.add_trace(go.Scattermapbox(
            lon=[row['centroid'].x],
            lat=[row['centroid'].y],
            text=row['region'],
            mode='text',
            showlegend=False,  # Hide the traces in the legend
            textfont=dict(size=10, color='black'),
        ))

    # Plot the interactive map
    st.plotly_chart(fig, use_container_width=True)

    # Add graphs description
    st.markdown("""
           <div style='text-align: center'>
               <strong>Please select a region to see the differences and changes in the graphs below.</strong>
           </div>
       """, unsafe_allow_html=True)

    # Select region based on dropdown
    selected_region = st.sidebar.selectbox("Select a region", merged.index, index=merged.index.tolist().index(st.session_state.selected_region))

    if selected_region:
        st.session_state.selected_region = selected_region  # Update session state
        region_data = preprocessed_flying_etiquette_df[preprocessed_flying_etiquette_df['Location'] == selected_region]

        # First graph: Politeness level by education
        education_politeness = region_data.groupby('Education')['politeness_score_normalized'].mean().reset_index()
        fig_education = px.bar(education_politeness, x='Education', y='politeness_score_normalized',
                               title=f'Politeness by Education in {selected_region}',
                               color_discrete_sequence=["#636EFA"])
        st.plotly_chart(fig_education, use_container_width=True)

        # Second graph: Politeness level by income level and gender
        income_gender_politeness = region_data.groupby(['Household Income', 'Gender'])['politeness_score_normalized'].mean().reset_index()
        fig_income_gender = px.bar(income_gender_politeness, x='Household Income', y='politeness_score_normalized', color='Gender', barmode='group',
                                   title=f'Politeness by Income Level and Gender in {selected_region}',
                                  )
        st.plotly_chart(fig_income_gender, use_container_width=True)
else:
    st.write("No data available for the selected filters.")

# Footer
st.markdown("""
    ---
    **Created by Ayelet Hashahar Cohen & Tzuf Lahan**  
""")
