import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# Page config
st.set_page_config(
    page_title="Spotify Global Streaming Dashboard",
    page_icon="🎵",
    layout="wide"
)

# Spotify colors
SPOTIFY_COLORS = {
    'background': '#191414',  # Spotify black
    'primary': '#1DB954',    # Spotify green
    'text': '#FFFFFF',       # White
    'secondary': '#535353'   # Dark gray
}

# Color palette for genres
GENRE_COLORS = {
    'Pop': '#1DB954',      # Spotify green
    'Hip Hop': '#1ed760',  # Lighter green
    'R&B': '#535353',      # Gray
    'Rock': '#b3b3b3',     # Light gray
    'Jazz': '#4687d6',     # Blue
    'Classical': '#ff6437', # Orange
    'EDM': '#ff5722',      # Deep orange
    'K-pop': '#8c1932',    # Red
    'Indie': '#af2896',    # Purple
    'Reggaeton': '#148a08' # Dark green
}

# Load and preprocess data
@st.cache_data
def load_data():
    df = pd.read_csv('Spotify_2024_Global_Streaming_Data.csv')
    return df

df = load_data()

# Initialize and train model
@st.cache_resource
def initialize_model():
    # Initialize label encoders
    country_le = LabelEncoder()
    artist_le = LabelEncoder()
    genre_le = LabelEncoder()
    
    # Fit label encoders
    df['Country_encoded'] = country_le.fit_transform(df['Country'])
    df['Artist_encoded'] = artist_le.fit_transform(df['Artist'])
    df['Genre_encoded'] = genre_le.fit_transform(df['Genre'])
    
    # Prepare features
    X = df[[
        'Country_encoded',
        'Artist_encoded',
        'Genre_encoded',
        'Release Year',
        'Monthly Listeners (Millions)',
        'Avg Stream Duration (Min)',
        'Skip Rate (%)'
    ]]
    y = df['Total Streams (Millions)']
    
    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    return model, country_le, artist_le, genre_le

model, country_le, artist_le, genre_le = initialize_model()

# Title
st.title("🎵 Spotify Global Streaming Analytics")

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "Global Trends", 
    "Top Artists", 
    "Geographic Distribution",
    "Stream Predictor"
])

# Tab 1: Global Trends
with tab1:
    st.header("Global Streaming Trends")
    
    genre_year_data = df.groupby(['Genre', 'Release Year'])['Total Streams (Millions)'].mean().reset_index()
    
    fig = go.Figure()
    
    for genre in df['Genre'].unique():
        genre_data = genre_year_data[genre_year_data['Genre'] == genre]
        fig.add_trace(go.Scatter(
            x=genre_data['Release Year'],
            y=genre_data['Total Streams (Millions)'],
            name=genre,
            line=dict(color=GENRE_COLORS.get(genre, '#1DB954'), width=2),
            mode='lines'
        ))
    
    fig.update_layout(
        template="plotly_dark",
        title="Streaming Trends by Genre",
        xaxis_title="Release Year",
        yaxis_title="Average Streams (Millions)",
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Tab 2: Top Artists
with tab2:
    st.header("Top Artists Analysis")
    
    top_artists = df.nlargest(10, 'Monthly Listeners (Millions)')
    
    fig = go.Figure(go.Bar(
        x=top_artists['Artist'],
        y=top_artists['Monthly Listeners (Millions)'],
        marker_color=SPOTIFY_COLORS['primary']
    ))
    
    fig.update_layout(
        template="plotly_dark",
        title="Top 10 Artists by Monthly Listeners",
        xaxis_title="Artist",
        yaxis_title="Monthly Listeners (Millions)",
        xaxis_tickangle=45
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Tab 3: Geographic Distribution
with tab3:
    st.header("Geographic Distribution")
    
    fig = go.Figure(data=go.Choropleth(
        locations=df['Country'],
        locationmode='country names',
        z=df['Total Streams (Millions)'],
        colorscale=[[0, SPOTIFY_COLORS['secondary']], [1, SPOTIFY_COLORS['primary']]],
        marker_line_color=SPOTIFY_COLORS['text'],
        marker_line_width=0.5
    ))
    
    fig.update_layout(
        template="plotly_dark",
        title="Global Streaming Distribution",
        geo=dict(showframe=False, showcoastlines=True)
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Tab 4: Stream Predictor
with tab4:
    st.header("Stream Predictor")
    st.write("Predict how many streams a song might get based on different factors:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        country = st.selectbox("1. Select Country", sorted(df['Country'].unique()))
        artist = st.selectbox("2. Choose Artist", sorted(df['Artist'].unique()))
        genre = st.selectbox("3. Select Genre", sorted(df['Genre'].unique()))
    
    with col2:
        release_year = st.number_input("4. Release Year", 
                                     min_value=2018, 
                                     max_value=2024, 
                                     value=2024)
        monthly_listeners = st.number_input("5. Monthly Listeners (Millions)", 
                                          min_value=0.0, 
                                          value=1.0, 
                                          step=0.1)
        stream_duration = st.number_input("6. Average Song Duration (Minutes)", 
                                        min_value=0.0, 
                                        value=3.0, 
                                        step=0.5)
        skip_rate = st.number_input("7. Skip Rate (%)", 
                                   min_value=0, 
                                   max_value=100, 
                                   value=20)
    
    if st.button("Calculate Streams", type="primary"):
        try:
            # Encode categorical variables
            country_encoded = country_le.transform([country])[0]
            artist_encoded = artist_le.transform([artist])[0]
            genre_encoded = genre_le.transform([genre])[0]
            
            # Prepare input data
            input_data = pd.DataFrame({
                'Country_encoded': [country_encoded],
                'Artist_encoded': [artist_encoded],
                'Genre_encoded': [genre_encoded],
                'Release Year': [release_year],
                'Monthly Listeners (Millions)': [monthly_listeners],
                'Avg Stream Duration (Min)': [stream_duration],
                'Skip Rate (%)': [skip_rate]
            })
            
            # Make prediction
            prediction = model.predict(input_data)[0]
            
            # Display result
            st.success(f"Predicted Total Streams: {prediction:.1f}M")
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>Created by <a href='https://github.com/hosseinmang'>@hosseinmang</a> | 
        Data source: Spotify Global Streaming Data 2024</p>
    </div>
    """,
    unsafe_allow_html=True
) 