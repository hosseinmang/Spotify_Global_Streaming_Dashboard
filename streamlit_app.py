import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from datetime import datetime

# Page config
st.set_page_config(
    page_title="Spotify Global Streaming Analytics",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with improved readability
st.markdown("""
<style>
    /* Main app background and text */
    .stApp {
        background-color: #191414 !important;
    }
    
    /* Main title */
    .main .block-container h1 {
        color: #1DB954 !important;
        font-weight: 600 !important;
        font-size: 48px !important;
        text-align: center !important;
        padding-bottom: 2rem !important;
    }
    
    /* Section headers */
    .main .block-container h2,
    .main .block-container h3,
    .main .block-container h4 {
        color: #1DB954 !important;
        font-weight: 500 !important;
        font-size: 28px !important;
        margin-bottom: 1rem !important;
    }
    
    /* Plotly chart styling */
    .js-plotly-plot .plotly .main-svg {
        background-color: #191414 !important;
    }
    
    .js-plotly-plot .plotly .bg {
        fill: #191414 !important;
    }
    
    /* Chart text */
    .js-plotly-plot .plotly .gtitle {
        fill: #FFFFFF !important;
        font-size: 24px !important;
        font-weight: 500 !important;
    }
    
    .js-plotly-plot .plotly .xtick text,
    .js-plotly-plot .plotly .ytick text {
        fill: #FFFFFF !important;
        font-size: 12px !important;
    }
    
    /* Grid lines - making them more subtle */
    .js-plotly-plot .plotly .gridlayer path {
        stroke: #282828 !important;
        stroke-opacity: 0.3 !important;
    }
    
    /* Axis lines */
    .js-plotly-plot .plotly .xaxis .zerolinelayer path,
    .js-plotly-plot .plotly .yaxis .zerolinelayer path {
        stroke: #535353 !important;
        stroke-width: 1 !important;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #191414;
        padding: 0;
        border-radius: 0;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #191414;
        border-radius: 0;
        padding: 10px 20px;
        color: #FFFFFF;
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #191414 !important;
        color: #1DB954 !important;
        font-weight: 600;
        border-bottom: 2px solid #1DB954 !important;
    }
    
    /* Inputs and selectboxes */
    .stSelectbox [data-baseweb="select"],
    .stNumberInput [data-baseweb="input"],
    .stMultiSelect [data-baseweb="select"] {
        background-color: #191414 !important;
        color: #FFFFFF !important;
        border: 1px solid #535353 !important;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #191414 !important;
        border-right: 1px solid #282828 !important;
    }
    
    [data-testid="stSidebar"] .block-container {
        padding-top: 2rem !important;
    }
    
    /* Text elements */
    .stMarkdown,
    .stText,
    p, div, span {
        color: #FFFFFF !important;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    div[data-testid="stToolbar"] {visibility: hidden;}
    
    /* Tooltip styling */
    .plotly .tooltip {
        background-color: #191414 !important;
        color: #FFFFFF !important;
        border: 1px solid #1DB954 !important;
    }
</style>
""", unsafe_allow_html=True)

# Update Spotify colors with additional shades
SPOTIFY_COLORS = {
    'background': '#191414',     # Spotify black
    'primary': '#1DB954',       # Spotify green
    'primary_light': '#1ed760', # Lighter green
    'text': '#FFFFFF',          # White
    'secondary': '#535353',     # Dark gray
    'surface': '#282828',       # Surface color
    'error': '#E91429',         # Error red
    'success': '#1DB954',       # Success green
    'warning': '#FF5722'        # Warning orange
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
    try:
        df = pd.read_csv('Spotify_2024_Global_Streaming_Data.csv')
        st.sidebar.success("✅ Data loaded successfully!")
        return df
    except Exception as e:
        st.sidebar.error(f"❌ Error loading data: {str(e)}")
        return None

df = load_data()

if df is None:
    st.error("Failed to load the dataset. Please check if the data file exists and is accessible.")
    st.stop()

# Initialize and train model
@st.cache_resource
def initialize_model():
    try:
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
        
        st.sidebar.success("✅ Model initialized successfully!")
        return model, country_le, artist_le, genre_le
    except Exception as e:
        st.sidebar.error(f"❌ Error initializing model: {str(e)}")
        return None, None, None, None

model, country_le, artist_le, genre_le = initialize_model()

if None in (model, country_le, artist_le, genre_le):
    st.error("Failed to initialize the model. Please check the error messages in the sidebar.")
    st.stop()

# Sidebar
with st.sidebar:
    st.image("https://storage.googleapis.com/pr-newsroom-wp/1/2018/11/Spotify_Logo_RGB_White.png", width=200)
    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    This dashboard analyzes Spotify's global streaming data and provides insights into:
    - Streaming trends by genre
    - Top performing artists
    - Geographic distribution
    - Stream predictions
    """)
    st.markdown("---")
    st.markdown("### Data Last Updated")
    st.markdown(f"_{datetime.now().strftime('%B %d, %Y')}_")

# Main content
st.title("🎵 Spotify Global Streaming Analytics")

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Global Trends", 
    "🎤 Top Artists", 
    "🌍 Geographic Distribution",
    "🔮 Stream Predictor"
])

# Tab 1: Global Trends
with tab1:
    st.header("Global Streaming Trends")
    
    # Time period selector
    col1, col2 = st.columns([2, 1])
    with col1:
        selected_genres = st.multiselect(
            "Select Genres to Display",
            options=sorted(df['Genre'].unique()),
            default=sorted(df['Genre'].unique())[:5]
        )
    
    with col2:
        metric_option = st.selectbox(
            "Select Metric",
            ["Total Streams (Millions)", "Monthly Listeners (Millions)", "Skip Rate (%)"]
        )
    
    # Filter data
    genre_year_data = df[df['Genre'].isin(selected_genres)].groupby(
        ['Genre', 'Release Year']
    )[metric_option].mean().reset_index()
    
    # Create visualization
    fig = px.line(
        genre_year_data,
        x='Release Year',
        y=metric_option,
        color='Genre',
        template="plotly_dark"
    )
    
    # Update layout directly
    fig.update_layout(
        title=dict(
            text=f"{metric_option} by Genre Over Time",
            font=dict(size=24, color='#FFFFFF'),
            x=0.5,
            y=0.95
        ),
        plot_bgcolor='#191414',
        paper_bgcolor='#191414',
        font_color='#FFFFFF',
        showlegend=True,
        legend=dict(
            bgcolor='#191414',
            font=dict(color='#FFFFFF'),
            bordercolor='#282828',
            borderwidth=1
        ),
        xaxis=dict(
            gridcolor='#282828',
            gridwidth=1,
            griddash='dot',
            tickfont=dict(color='#FFFFFF'),
            showline=True,
            linecolor='#535353',
            linewidth=1
        ),
        yaxis=dict(
            gridcolor='#282828',
            gridwidth=1,
            griddash='dot',
            tickfont=dict(color='#FFFFFF'),
            showline=True,
            linecolor='#535353',
            linewidth=1
        ),
        margin=dict(t=100, b=50, l=50, r=50)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Summary statistics
    st.subheader("Summary Statistics")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        avg_streams = df[df['Genre'].isin(selected_genres)]['Total Streams (Millions)'].mean()
        st.metric("Average Streams (M)", f"{avg_streams:.1f}")
    
    with col2:
        avg_listeners = df[df['Genre'].isin(selected_genres)]['Monthly Listeners (Millions)'].mean()
        st.metric("Average Monthly Listeners (M)", f"{avg_listeners:.1f}")
    
    with col3:
        avg_skip = df[df['Genre'].isin(selected_genres)]['Skip Rate (%)'].mean()
        st.metric("Average Skip Rate", f"{avg_skip:.1f}%")

# Tab 2: Top Artists
with tab2:
    st.header("Top Artists Analysis")
    
    metric = st.selectbox(
        "Select Ranking Metric",
        ["Monthly Listeners (Millions)", "Total Streams (Millions)"]
    )
    
    top_n = st.slider("Number of Artists to Display", 5, 20, 10)
    
    top_artists = df.nlargest(top_n, metric)
    
    fig = px.bar(
        top_artists,
        x='Artist',
        y=metric,
        color='Genre',
        template="plotly_dark"
    )
    
    fig.update_layout(
        title=dict(
            text=f"Top {top_n} Artists by {metric}",
            font=dict(size=24, color='#FFFFFF'),
            x=0.5,
            y=0.95
        ),
        plot_bgcolor='#191414',
        paper_bgcolor='#191414',
        font_color='#FFFFFF',
        showlegend=True,
        legend=dict(
            bgcolor='#191414',
            font=dict(color='#FFFFFF'),
            bordercolor='#282828',
            borderwidth=1
        ),
        xaxis=dict(
            gridcolor='#282828',
            gridwidth=1,
            griddash='dot',
            tickfont=dict(color='#FFFFFF'),
            showline=True,
            linecolor='#535353',
            linewidth=1,
            tickangle=45
        ),
        yaxis=dict(
            gridcolor='#282828',
            gridwidth=1,
            griddash='dot',
            tickfont=dict(color='#FFFFFF'),
            showline=True,
            linecolor='#535353',
            linewidth=1
        ),
        margin=dict(t=100, b=100, l=50, r=50)  # Increased bottom margin for rotated labels
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Artist details
    if st.checkbox("Show Detailed Artist Statistics"):
        st.dataframe(
            top_artists[['Artist', 'Genre', 'Monthly Listeners (Millions)', 
                        'Total Streams (Millions)', 'Skip Rate (%)']]
            .style.background_gradient(cmap='Greens')
        )

# Tab 3: Geographic Distribution
with tab3:
    st.header("Geographic Distribution")
    
    metric = st.selectbox(
        "Select Geographic Metric",
        ["Total Streams (Millions)", "Monthly Listeners (Millions)", "Skip Rate (%)"],
        key="geo_metric"
    )
    
    # Aggregate data by country
    country_data = df.groupby('Country')[metric].mean().reset_index()
    
    fig = px.choropleth(
        country_data,
        locations='Country',
        locationmode='country names',
        color=metric,
        color_continuous_scale=['#535353', '#1DB954'],
        template="plotly_dark"
    )
    
    fig.update_layout(
        title=dict(
            text=f"Global Distribution of {metric}",
            font=dict(size=24, color='#FFFFFF'),
            x=0.5,
            y=0.95
        ),
        plot_bgcolor='#191414',
        paper_bgcolor='#191414',
        font_color='#FFFFFF',
        showlegend=True,
        legend=dict(
            bgcolor='#191414',
            font=dict(color='#FFFFFF'),
            bordercolor='#282828',
            borderwidth=1
        ),
        margin=dict(t=100, b=50, l=50, r=50),
        geo=dict(
            showframe=False,
            showcoastlines=True,
            projection_type='equirectangular',
            bgcolor='#191414',
            lakecolor='#191414',
            landcolor='#282828',
            coastlinecolor='#535353',
            countrycolor='#535353'
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Country rankings
    st.subheader("Country Rankings")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Top 5 Countries")
        st.dataframe(country_data.nlargest(5, metric))
    
    with col2:
        st.markdown("#### Bottom 5 Countries")
        st.dataframe(country_data.nsmallest(5, metric))

# Tab 4: Stream Predictor
with tab4:
    st.header("Stream Predictor")
    st.write("Predict potential streams based on various factors:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        country = st.selectbox("1. Select Country", sorted(df['Country'].unique()))
        artist = st.selectbox("2. Choose Artist", sorted(df['Artist'].unique()))
        genre = st.selectbox("3. Select Genre", sorted(df['Genre'].unique()))
    
    with col2:
        release_year = st.number_input(
            "4. Release Year",
            min_value=2018,
            max_value=2024,
            value=2024,
            help="Year of song release (2018-2024)"
        )
        
        monthly_listeners = st.number_input(
            "5. Monthly Listeners (Millions)",
            min_value=0.0,
            value=1.0,
            step=0.1,
            help="Average monthly listeners in millions"
        )
        
        stream_duration = st.number_input(
            "6. Average Song Duration (Minutes)",
            min_value=0.0,
            value=3.0,
            step=0.5,
            help="Average length of songs in minutes"
        )
        
        skip_rate = st.number_input(
            "7. Skip Rate (%)",
            min_value=0,
            max_value=100,
            value=20,
            help="Percentage of times songs are skipped"
        )
    
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
            
            # Create columns for the result display
            col1, col2, col3 = st.columns([1, 2, 1])
            
            with col2:
                st.markdown(
                    f"""
                    <div style='
                        background-color: {SPOTIFY_COLORS['surface']};
                        padding: 24px;
                        border-radius: 15px;
                        border: 2px solid {SPOTIFY_COLORS['primary']};
                        text-align: center;
                        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
                    '>
                        <h3 style='color: {SPOTIFY_COLORS['text']}; margin: 0; font-size: 20px;'>
                            Predicted Total Streams
                        </h3>
                        <h2 style='
                            color: {SPOTIFY_COLORS['primary']};
                            margin: 15px 0;
                            font-size: 36px;
                            font-weight: bold;
                        '>
                            {prediction:.1f}M
                        </h2>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            
            # Show feature importance
            st.markdown("### Feature Importance")
            feature_importance = pd.DataFrame({
                'Feature': ['Country', 'Artist', 'Genre', 'Release Year', 
                           'Monthly Listeners', 'Stream Duration', 'Skip Rate'],
                'Importance': model.feature_importances_
            })
            
            fig = px.bar(
                feature_importance.sort_values('Importance', ascending=True),
                x='Importance',
                y='Feature',
                orientation='h',
                template="plotly_dark"
            )
            
            fig.update_layout(
                title=dict(
                    text="Feature Importance in Prediction",
                    font=dict(size=24, color='#FFFFFF'),
                    x=0.5,
                    y=0.95
                ),
                plot_bgcolor='#191414',
                paper_bgcolor='#191414',
                font_color='#FFFFFF',
                showlegend=False,
                xaxis=dict(
                    gridcolor='#282828',
                    gridwidth=1,
                    griddash='dot',
                    tickfont=dict(color='#FFFFFF'),
                    showline=True,
                    linecolor='#535353',
                    linewidth=1
                ),
                yaxis=dict(
                    gridcolor='#282828',
                    gridwidth=1,
                    griddash='dot',
                    tickfont=dict(color='#FFFFFF'),
                    showline=True,
                    linecolor='#535353',
                    linewidth=1
                ),
                margin=dict(t=100, b=50, l=150, r=50)  # Increased left margin for feature names
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

# Footer
st.markdown("---")
st.markdown(
    f"""
    <div style='
        text-align: center;
        padding: 20px;
        background-color: {SPOTIFY_COLORS['surface']};
        border-radius: 10px;
        margin-top: 30px;
    '>
        <p style='color: {SPOTIFY_COLORS['text']}; margin: 0;'>
            Created by <a href='https://github.com/hosseinmang'>@hosseinmang</a> | 
            Data source: Spotify Global Streaming Data 2024
        </p>
    </div>
    """,
    unsafe_allow_html=True
) 