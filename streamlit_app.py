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

# Custom CSS with modern design
st.markdown("""
<style>
    /* Main app styling */
    .stApp {
        background-color: #FAFAFA !important;
    }
    
    /* Main title */
    .main .block-container h1 {
        color: #191414 !important;
        font-weight: 700 !important;
        font-size: 42px !important;
        text-align: center !important;
        padding: 1.5rem 0 !important;
        margin-bottom: 2rem !important;
        border-bottom: 3px solid #1DB954 !important;
    }
    
    /* Section headers */
    .main .block-container h2 {
        color: #191414 !important;
        font-weight: 600 !important;
        font-size: 32px !important;
        margin: 1rem 0 !important;
        padding-left: 1rem !important;
        border-left: 4px solid #1DB954 !important;
    }
    
    .main .block-container h3 {
        color: #191414 !important;
        font-weight: 500 !important;
        font-size: 24px !important;
        margin: 0.5rem 0 !important;
    }
    
    /* Card styling */
    div[data-testid="stVerticalBlock"] > div {
        background-color: #FFFFFF;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.05);
        margin-bottom: 1rem;
    }
    
    /* Metric styling */
    div[data-testid="metric-container"] {
        background-color: #FFFFFF;
        border: 1px solid #E5E5E5;
        border-radius: 8px;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.02);
    }
    
    div[data-testid="metric-container"] label {
        color: #666666 !important;
    }
    
    div[data-testid="metric-container"] div[data-testid="metric-value"] {
        color: #1DB954 !important;
        font-weight: 600 !important;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 16px;
        background-color: #FFFFFF;
        padding: 16px 24px;
        border-radius: 10px;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.05);
        margin-bottom: 2rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: transparent;
        border: none;
        color: #666666;
        font-weight: 500;
        padding: 8px 24px;
        border-radius: 25px;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #1DB954 !important;
        color: #FFFFFF !important;
        font-weight: 600;
    }
    
    /* Input fields styling */
    .stSelectbox [data-baseweb="select"],
    .stNumberInput [data-baseweb="input"],
    .stMultiSelect [data-baseweb="select"] {
        background-color: #FFFFFF;
        border: 2px solid #E5E5E5;
        border-radius: 8px;
        transition: all 0.3s ease;
    }
    
    .stSelectbox [data-baseweb="select"]:hover,
    .stNumberInput [data-baseweb="input"]:hover,
    .stMultiSelect [data-baseweb="select"]:hover {
        border-color: #1DB954;
    }
    
    /* Button styling */
    .stButton > button {
        background-color: #1DB954;
        color: #FFFFFF;
        font-weight: 600;
        padding: 0.5rem 2rem;
        border-radius: 25px;
        border: none;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background-color: #1ed760;
        box-shadow: 0 4px 12px rgba(29, 185, 84, 0.3);
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #FFFFFF;
        border-right: 1px solid #E5E5E5;
        padding: 2rem 1rem;
    }
    
    [data-testid="stSidebar"] .block-container {
        margin-top: 2rem;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    div[data-testid="stToolbar"] {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Update Spotify colors with additional shades
SPOTIFY_COLORS = {
    'background': '#FFFFFF',     # White background
    'primary': '#1DB954',       # Spotify green
    'primary_light': '#1ed760', # Lighter green
    'text': '#191414',          # Black text
    'secondary': '#535353',     # Dark gray
    'surface': '#F8F8F8',       # Light surface
    'error': '#E91429',         # Error red
    'success': '#1DB954',       # Success green
    'warning': '#FF5722'        # Warning orange
}

# Color palette for genres with more distinct colors
GENRE_COLORS = {
    'Pop': '#1DB954',      # Spotify green
    'Hip Hop': '#FF6B6B',  # Coral red
    'R&B': '#4A90E2',      # Blue
    'Rock': '#FFD93D',     # Yellow
    'Jazz': '#FF8C42',     # Orange
    'Classical': '#6C5B7B', # Purple
    'EDM': '#45B7D1',      # Cyan
    'K-pop': '#FF4081',    # Pink
    'Indie': '#2ECC71',    # Emerald
    'Reggaeton': '#F39C12' # Golden
}

# Update all plot layouts to use the new theme
def update_plot_theme(fig):
    fig.update_layout(
        plot_bgcolor='#FFFFFF',
        paper_bgcolor='#FFFFFF',
        font_color='#191414',
        title=dict(
            font=dict(color='#191414', size=24),
            x=0.5,
            y=0.95
        ),
        legend=dict(
            bgcolor='#FFFFFF',
            font=dict(color='#191414'),
            bordercolor='#E5E5E5',
            borderwidth=1
        ),
        xaxis=dict(
            gridcolor='#E5E5E5',
            gridwidth=1,
            griddash='dot',
            tickfont=dict(color='#191414'),
            showline=True,
            linecolor='#535353',
            linewidth=1,
            title_font=dict(color='#191414', size=14)
        ),
        yaxis=dict(
            gridcolor='#E5E5E5',
            gridwidth=1,
            griddash='dot',
            tickfont=dict(color='#191414'),
            showline=True,
            linecolor='#535353',
            linewidth=1,
            title_font=dict(color='#191414', size=14)
        ),
        hoverlabel=dict(
            bgcolor='#FFFFFF',
            font_size=14,
            font_color='#191414'
        )
    )
    return fig

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

# Sidebar with modern design
with st.sidebar:
    st.image("https://storage.googleapis.com/pr-newsroom-wp/1/2018/11/Spotify_Logo_RGB_Green.png", width=180)
    st.markdown("---")
    
    st.markdown("""
    <div style='
        background-color: #F8F9FA;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    '>
        <h4 style='color: #191414; margin-bottom: 10px;'>About</h4>
        <p style='color: #666666; font-size: 14px; line-height: 1.6;'>
            Explore Spotify's global streaming insights through:
        </p>
        <ul style='color: #666666; font-size: 14px; line-height: 1.6; padding-left: 20px;'>
            <li>Streaming trends analysis</li>
            <li>Top artist performance</li>
            <li>Geographic distribution</li>
            <li>AI-powered predictions</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown(f"""
    <div style='
        background-color: #F8F9FA;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
    '>
        <p style='color: #666666; font-size: 12px; margin: 0;'>
            Last Updated<br>
            <span style='color: #1DB954; font-weight: 500;'>
                {datetime.now().strftime('%B %d, %Y')}
            </span>
        </p>
    </div>
    """, unsafe_allow_html=True)

# Main content with modern layout
st.title("🎵 Spotify Global Streaming Analytics")

# Create modern tabs
tabs = st.tabs([
    "📈  Trends",
    "🎤  Artists",
    "🌍  Geography",
    "🔮  Predictor"
])

# Tab 1: Global Trends
with tabs[0]:
    st.header("Global Streaming Trends")
    
    # Modern filter section
    with st.container():
        st.markdown("""
        <div style='
            background-color: #F8F9FA;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
        '>
            <h4 style='color: #191414; margin-bottom: 15px;'>Filter Options</h4>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        with col1:
            selected_genres = st.multiselect(
                "Select Genres",
                options=sorted(df['Genre'].unique()),
                default=sorted(df['Genre'].unique())[:5]
            )
        
        with col2:
            metric_option = st.selectbox(
                "Metric",
                ["Total Streams (Millions)", "Monthly Listeners (Millions)", "Skip Rate (%)"]
            )
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Create visualization with enhanced styling
    if selected_genres:
        genre_year_data = df[df['Genre'].isin(selected_genres)].groupby(
            ['Genre', 'Release Year']
        )[metric_option].mean().reset_index()
        
        fig = px.line(
            genre_year_data,
            x='Release Year',
            y=metric_option,
            color='Genre',
            template="none",
            color_discrete_map=GENRE_COLORS
        )
        
        fig.update_traces(
            line=dict(width=3),
            mode='lines+markers',
            marker=dict(size=8)
        )
        
        fig.update_layout(
            plot_bgcolor='#FFFFFF',
            paper_bgcolor='#FFFFFF',
            font_color='#191414',
            title=dict(
                text=f"{metric_option} by Genre Over Time",
                font=dict(size=24, color='#191414'),
                x=0.5,
                y=0.95
            ),
            legend=dict(
                bgcolor='#FFFFFF',
                bordercolor='#E5E5E5',
                borderwidth=1
            ),
            xaxis=dict(
                gridcolor='#E5E5E5',
                showline=True,
                linecolor='#E5E5E5',
                title_font=dict(size=14)
            ),
            yaxis=dict(
                gridcolor='#E5E5E5',
                showline=True,
                linecolor='#E5E5E5',
                title_font=dict(size=14)
            ),
            margin=dict(t=80, b=40, l=40, r=40)
        )
        
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        
        # Summary statistics with modern cards
        st.subheader("Key Metrics")
        
        metrics_container = st.container()
        with metrics_container:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                avg_streams = df[df['Genre'].isin(selected_genres)]['Total Streams (Millions)'].mean()
                st.metric(
                    "Average Streams",
                    f"{avg_streams:.1f}M",
                    delta=None
                )
            
            with col2:
                avg_listeners = df[df['Genre'].isin(selected_genres)]['Monthly Listeners (Millions)'].mean()
                st.metric(
                    "Monthly Listeners",
                    f"{avg_listeners:.1f}M",
                    delta=None
                )
            
            with col3:
                avg_skip = df[df['Genre'].isin(selected_genres)]['Skip Rate (%)'].mean()
                st.metric(
                    "Skip Rate",
                    f"{avg_skip:.1f}%",
                    delta=None
                )
    else:
        st.warning("Please select at least one genre to display the trends.")

# Tab 2: Top Artists
with tabs[1]:
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
        template="none",
        color_discrete_map=GENRE_COLORS,
        barmode='group'
    )
    
    update_plot_theme(fig)
    fig.update_traces(
        marker_line_color='#FFFFFF',
        marker_line_width=1.5,
        opacity=0.8
    )
    fig.update_layout(
        title=dict(
            text=f"Top {top_n} Artists by {metric}",
            font=dict(size=24, color='#191414'),
            x=0.5,
            y=0.95
        ),
        xaxis_tickangle=45,
        margin=dict(t=100, b=100, l=50, r=50),
        showlegend=True
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
with tabs[2]:
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
        color_continuous_scale=[[0, '#E5E5E5'], [0.5, '#9BE5B9'], [1, '#1DB954']],
        template="none"
    )
    
    update_plot_theme(fig)
    fig.update_layout(
        title=dict(
            text=f"Global Distribution of {metric}",
            font=dict(size=24, color='#191414'),
            x=0.5,
            y=0.95
        ),
        margin=dict(t=100, b=50, l=50, r=50),
        geo=dict(
            showframe=False,
            showcoastlines=True,
            projection_type='equirectangular',
            bgcolor='#FFFFFF',
            lakecolor='#FFFFFF',
            landcolor='#F8F8F8',
            coastlinecolor='#535353',
            countrycolor='#E5E5E5',
            showocean=True,
            oceancolor='#FFFFFF'
        ),
        coloraxis_colorbar=dict(
            title=metric,
            tickfont=dict(color='#191414'),
            titlefont=dict(color='#191414')
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
with tabs[3]:
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
                template="none",
                color_discrete_sequence=['#1DB954']
            )
            
            update_plot_theme(fig)
            fig.update_traces(
                marker_line_color='#FFFFFF',
                marker_line_width=1,
                opacity=0.8
            )
            fig.update_layout(
                title=dict(
                    text="Feature Importance in Prediction",
                    font=dict(size=24, color='#191414'),
                    x=0.5,
                    y=0.95
                ),
                margin=dict(t=100, b=50, l=150, r=50),
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

# Footer with modern design
st.markdown("---")
st.markdown(
    f"""
    <div style='
        text-align: center;
        padding: 20px;
        background-color: #F8F9FA;
        border-radius: 10px;
        margin-top: 30px;
    '>
        <p style='color: #666666; font-size: 14px; margin: 0;'>
            Developed by <a href='https://github.com/hosseinmang' style='color: #1DB954; text-decoration: none;'>@hosseinmang</a> | 
            Data: Spotify Global Streaming 2024
        </p>
    </div>
    """,
    unsafe_allow_html=True
) 