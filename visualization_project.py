import json
import requests
import numpy as np
import streamlit as st
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from matplotlib import cm
from shapely.geometry import Polygon


def read_csv_from_googledrive(url):
    url = 'https://drive.google.com/uc?id=' + url.split('/')[-2]
    df = pd.read_csv(url)
    return df

# Load the preprocessed DataFrame
preprocessed_flying_etiquette_df = read_csv_from_googledrive('https://drive.google.com/file/d/1H0tBXnR_tYa0pvzVi1XI5mESRDr7Q96j/view?usp=sharing')

# Define available options for filters
income_levels = ['All', '$0 - $24,999', '$25,000 - $49,999', '$50,000 - $99,999', '$100,000 - $149,999', '150000']
genders = ['All', 'Male', 'Female']
Region_list = ['All', 'Pacific', 'Mountain', 'West North Central', 'West South Central', 'East North Central', 'East South Central', 'South Atlantic', 'Middle Atlantic', 'New England']


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
    else:
        continue
    # Add other regions as needed
    new_region = gpd.GeoDataFrame({'region': [region], 'geometry': [polygon]})
    gdf = gdf.append(new_region, ignore_index=True)

# Calculate the global min and max for the normalized politeness score
politeness_score_by_filters = []
for selected_income in income_levels:
    for selected_gender in genders:
        for selected_region in Region_list:
            if selected_region == 'All':
                filtered_data = preprocessed_flying_etiquette_df[
                    (preprocessed_flying_etiquette_df['Household Income'] == selected_income) &
                    (preprocessed_flying_etiquette_df['Gender'] == selected_gender)
                ]
            else:
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
    st.session_state.selected_region = 'All'  # Default region

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
    <div style='text-align: left; font-size: 17px;'>
        Welcome to the US Regions Politeness Heat Map! This interactive app allows you to explore the average politeness scores 
        across different regions in the United States based on household income and gender.
        <br><br>
        Data was collected from people in different areas across the United States, these people were asked about different situations and interactions between strangers on flights and asked their opinion on the subject 
        We used these findings to build a politeness level for each person and below are our results
    </div>
""", unsafe_allow_html=True)


# Add map description
st.markdown("""
    <div style='text-align: center; font-size: 25px;'>
        <br><br>
        <strong>Please select Household Income and Gender to see the politeness differences between various regions in the United States.</strong>
    </div>
""", unsafe_allow_html=True)


# Filter data based on selections
selected_income = st.selectbox('Select Household Income', income_levels)
selected_gender = st.selectbox('Select Gender', genders)

# Filter data based on selections
filtered_data = preprocessed_flying_etiquette_df[
    ((preprocessed_flying_etiquette_df['Household Income'] == selected_income) | (selected_income == 'All')) &
    ((preprocessed_flying_etiquette_df['Gender'] == selected_gender) | (selected_gender == 'All'))
]

# Check if the filtered data is not empty
if not filtered_data.empty:
    # Group by region and calculate the average normalized politeness score
    region_politeness = filtered_data.groupby('Mapped Location')['politeness_score_normalized'].mean().reset_index()
    region_politeness.columns = ['region', 'avg_politeness_score_normalized']

    # Define a distinct blue color palette for bins
    blues_cmap = px.colors.sequential.Blues[2:8]

    # Merge the data with the shapefile
    merged = gdf.set_index('region').join(region_politeness.set_index('region'))

    # Calculate the bins and assign colors
    merged['bins'] = pd.qcut(merged['avg_politeness_score_normalized'], 6, labels=blues_cmap)
    merged['avg_politeness_score_normalized'] = merged['avg_politeness_score_normalized'].round(2)

    # Convert GeoDataFrame to GeoJSON
    merged_geojson = json.loads(merged.to_json())

    # Add a manual colorbar
    colorbar_ticks = np.linspace(merged['avg_politeness_score_normalized'].min(),
                                 merged['avg_politeness_score_normalized'].max(), 6)
    colorbar_labels = [f"{v:.2f}" for v in colorbar_ticks]

    # Create an interactive map with Plotly
    fig = px.choropleth_mapbox(
        merged,
        geojson=merged_geojson,
        locations=merged.index,
        color='avg_politeness_score_normalized',
        color_continuous_scale=blues_cmap,
        mapbox_style="open-street-map",
        zoom=3,
        center={"lat": 37.0902, "lon": -95.7129},
        opacity=0.5,
        labels={'avg_politeness_score_normalized': 'Avg Politeness Score'},
        hover_data={'avg_politeness_score_normalized': True}
    )

    fig.update_layout(coloraxis_colorbar=dict(
        title="Avg Politeness Score",
        tickvals=colorbar_ticks,
        ticktext=colorbar_labels
    ))

    # Calculate centroids for each region to place the text annotations
    gdf['centroid'] = gdf.geometry.centroid
    for idx, row in gdf.iterrows():
        fig.add_trace(go.Scattermapbox(
            lon=[row['centroid'].x],
            lat=[row['centroid'].y],
            text=row['region'],
            mode='text',
            showlegend=False,  # Hide the traces in the legend
            textfont=dict(size=10, color='black'),
        ))

    # Add title to the map
    fig.update_layout(
        title_text='Politeness Map by Regions in the USA',
        title_x=0
    )

    # Plot the interactive map
    st.plotly_chart(fig, use_container_width=True)

    
   # Add graphs description
    st.markdown("""
           <div style='text-align: center; font-size: 25px;'>
               <strong>Please select a region to see the differences and changes in the graphs below.</strong>
           </div>
       """, unsafe_allow_html=True)
    
    # Select region based on dropdown
    selected_region = st.selectbox("Select a region", Region_list, index=Region_list.index(st.session_state.selected_region))
    
    if selected_region:
        st.session_state.selected_region = selected_region  # Update session state
        if selected_region != 'All':
            region_data = preprocessed_flying_etiquette_df[preprocessed_flying_etiquette_df['Location'] == selected_region]
        else:
            region_data = preprocessed_flying_etiquette_df
    
        # Group by education and calculate politeness and population size
        education_politeness = region_data.groupby('Education').agg(
            politeness_score_normalized=('politeness_score_normalized', 'mean'),
            population_size=('politeness_score_normalized', 'size')
        ).reset_index()

    
        # Group by income level and gender
        income_gender_politeness = region_data.groupby(['Household Income', 'Gender'])['politeness_score_normalized'].mean().reset_index()
    
        # Create two columns in the Streamlit app
        col1, col2 = st.columns([1.5, 1])
    
        with col1:
            # First graph: Politeness level by education
            # Define the logical order for the education levels
            education_order = [
                'Less than high school', 'High school degree', 'Some college or Associate degree', 'Bachelor degree', 'Graduate degree'
            ]
            
            # Ensure 'Education' column is categorical with the defined order
            region_data['Education'] = pd.Categorical(region_data['Education'], categories=education_order, ordered=True)
            
            # Re-plot the graph with the ordered education levels
            fig_education = px.scatter(
                education_politeness, 
                x='Education', 
                y='politeness_score_normalized',
                size='population_size', 
                color='Education',
                title=f'Politeness by Education in {selected_region}',
                category_orders={"Education": education_order},  # Order categories
                color_discrete_sequence=px.colors.qualitative.Set1, #px.colors.sequential.Blues[::-1],  # Adjust color sequence for deeper colors
                size_max=40,  
                range_y=[0, education_politeness['politeness_score_normalized'].max() * 1.4]  
            )

            fig_education.update_traces(marker=dict(sizemin=10))  # Ensure smallest bubble is still visible
            fig_education.update_layout(margin=dict(t=50, b=100, l=50, r=50))
            st.plotly_chart(fig_education, use_container_width=True)
    
        with col2:
            # Second graph: Politeness level by income level and gender
            # Define the logical order for the income levels
            income_order = [
                '$0 - $24,999', '$25,000 - $49,999', '$50,000 - $99,999',
                '$100,000 - $149,999', '150000'
            ]
            
            # Ensure 'Household Income' column is categorical with the defined order
            region_data['Household Income'] = pd.Categorical(region_data['Household Income'], categories=income_order, ordered=True)
            
            # Re-plot the bar chart with the ordered income levels
            fig_income_gender = px.bar(
                income_gender_politeness, 
                x='Household Income', 
                y='politeness_score_normalized', 
                color='Gender', 
                barmode='group',
                title=f'Politeness by Income Level and Gender in {selected_region}',
                category_orders={"Household Income": income_order},  # Order categories
                color_discrete_map={'Female': '#aec7e8', 'Male': '#1f77b4'}
            )

            # fig_income_gender = px.bar(income_gender_politeness, x='Household Income', y='politeness_score_normalized', color='Gender', barmode='group',
            #                            title=f'Politeness by Income Level and Gender in {selected_region}', color_discrete_map={'Female': '#aec7e8', 'Male': '#1f77b4'}
            #                           )
            fig_income_gender.update_layout(margin=dict(t=50, b=100, l=50, r=50))
            st.plotly_chart(fig_income_gender, use_container_width=True, width=200, height=50)



# Group data by height and calculate politeness and population size
height_politeness = region_data.groupby('Height').agg(
    politeness_score_normalized=('politeness_score_normalized', 'mean'),
    population_size=('politeness_score_normalized', 'size')
).reset_index()

# Plot the scatter plot for Height vs Politeness
fig_height = px.scatter(
    height_politeness, 
    x='Height', 
    y='politeness_score_normalized',
    size='population_size', 
    color='Height',
    title=f'Politeness by Height in {selected_region}',
    size_max=40,  
    range_y=[0, height_politeness['politeness_score_normalized'].max() * 1.4]
)

fig_height.update_traces(marker=dict(sizemin=10))  # Ensure smallest bubble is still visible
fig_height.update_layout(margin=dict(t=50, b=100, l=50, r=50))
st.plotly_chart(fig_height, use_container_width=True)


# Define the order you want for the 'Seat Recline Frequency'
recline_order = ['Never', 'Once in a while', 'About half the time', 'Usually', 'Always']

# Ensure 'Seat Recline Frequency' column is categorical with the defined order
region_data['Seat Recline Frequency'] = pd.Categorical(region_data['Seat Recline Frequency'], categories=recline_order, ordered=True)

# Group data by seat recline frequency and calculate politeness score
seat_recline_politeness = region_data.groupby('Seat Recline Frequency').agg(
    politeness_score_normalized=('politeness_score_normalized', 'mean'),
    population_size=('politeness_score_normalized', 'size')
).reset_index()

# Define bins for politeness score
bins = [0, 0.33, 0.66, 1.0]
labels = ['Low', 'Medium', 'High']

# Create a new column for politeness level range based on the bins
region_data['Politeness_Level_Range'] = pd.cut(region_data['politeness_score_normalized'], bins=bins, labels=labels)

# Group data by seat recline frequency and politeness level range
seat_recline_politeness = region_data.groupby(['Seat Recline Frequency', 'Politeness_Level_Range']).size().reset_index(name='count')

# Map recline frequencies and politeness level ranges to numerical indices
recline_mapping = {level: idx for idx, level in enumerate(recline_order)}
politeness_mapping = {level: idx + len(recline_mapping) for idx, level in enumerate(seat_recline_politeness['Politeness_Level_Range'].unique())}

# Create source and target lists
source = seat_recline_politeness['Seat Recline Frequency'].map(recline_mapping).tolist()
target = seat_recline_politeness['Politeness_Level_Range'].map(politeness_mapping).tolist()
value = seat_recline_politeness['count'].tolist()

# Define darker color mapping based on the colors in the provided image
recline_colors = {
    'Never': '#A34B61',  
    'Once in a while': '#D18A3D',  
    'Always': '#4F8B4E',  
    'Usually': '#6A3D82',  
    'About half the time': '#2C6EA8'  
}

# Define darker color mapping for the Politeness Level Ranges
politeness_colors = {
    'Low': '#696969',
    'Medium': '#C4743D',
    'High': '#206060' 
}

# Assign colors to the nodes
node_colors = (
    [recline_colors.get(label, 'darkgray') for label in recline_order] + 
    [politeness_colors.get(label, 'darkgray') for label in politeness_mapping.keys()]
)

# Define color mapping for links based on the target
link_colors = []
for target_value in target:
    if target_value == politeness_mapping['Low']:
        link_colors.append('#7FB3D5')  # Light blue for Low
    elif target_value == politeness_mapping['Medium']:
        link_colors.append('#5DADE2')  # Medium blue for Medium
    elif target_value == politeness_mapping['High']:
        link_colors.append('#1F77B4')  # Darker blue for High

# Generate the Sankey diagram with custom colors for links
fig_sankey = go.Figure(go.Sankey(
    node=dict(
        pad=15,
        thickness=30,
        line=dict(color="gray", width=0.5),
        label=list(recline_order) + list(politeness_mapping.keys()),
        color=node_colors
    ),
    link=dict(
        source=source,
        target=target,
        value=value,
        color=link_colors  # Apply the custom link colors
    )
))

fig_sankey.update_layout(
    title_text="Politeness by Seat Recline Frequency",
    font_size=14,
    title_font_size=24,
    font_color='white'  
)
st.plotly_chart(fig_sankey, use_container_width=True)

# Plot a violin plot for politeness distribution by gender
fig_violin = px.violin(
    region_data, 
    y="politeness_score_normalized", 
    x="Gender", 
    color="Gender",
    box=True, 
    points="all",  # Show all points in the distribution
    title="Politeness Distribution by Gender",
    color_discrete_map={'Female': '#FF69B4', 'Male': '#1f77b4'}
)
# Display the violin plot
st.plotly_chart(fig_violin, use_container_width=True)

# Footer
st.markdown("""
    ---
    **Created by Ayelet Hashahar Cohen & Tzuf Lahan**  
""")
