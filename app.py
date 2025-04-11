import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import joblib

# Load and preprocess data
df = pd.read_csv('Spotify_2024_Global_Streaming_Data.csv')

# Initialize label encoders for categorical variables
country_le = LabelEncoder()
artist_le = LabelEncoder()
genre_le = LabelEncoder()

# Fit label encoders
df['Country_encoded'] = country_le.fit_transform(df['Country'])
df['Artist_encoded'] = artist_le.fit_transform(df['Artist'])
df['Genre_encoded'] = genre_le.fit_transform(df['Genre'])

# Prepare features for model training
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

# Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Save the model and encoders
joblib.dump(model, 'spotify_model.joblib')
joblib.dump(country_le, 'country_encoder.joblib')
joblib.dump(artist_le, 'artist_encoder.joblib')
joblib.dump(genre_le, 'genre_encoder.joblib')

# Spotify color theme
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

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])

# Create streaming trends figure
def create_streaming_trends():
    genre_year_data = df.groupby(['Genre', 'Release Year'])['Total Streams (Millions)'].mean().reset_index()
    
    fig = go.Figure()
    
    for genre in df['Genre'].unique():
        genre_data = genre_year_data[genre_year_data['Genre'] == genre]
        fig.add_trace(go.Scatter(
            x=genre_data['Release Year'],
            y=genre_data['Total Streams (Millions)'],
            name=genre,
            line=dict(color=GENRE_COLORS.get(genre, '#1DB954'), width=2),
            mode='lines',
            hovertemplate="<b>%{x}</b><br>" +
                         "Genre: " + genre + "<br>" +
                         "Streams: %{y:.0f}M<br>" +
                         "<extra></extra>"
        ))
    
    fig.update_layout(
        plot_bgcolor=SPOTIFY_COLORS['background'],
        paper_bgcolor=SPOTIFY_COLORS['background'],
        font_color=SPOTIFY_COLORS['text'],
        title={
            'text': 'Streaming Trends by Genre',
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'size': 24}
        },
        xaxis=dict(
            title='Release Year',
            gridcolor=SPOTIFY_COLORS['secondary'],
            gridwidth=0.5,
            showline=True,
            linecolor=SPOTIFY_COLORS['secondary']
        ),
        yaxis=dict(
            title='Total Streams (Millions)',
            gridcolor=SPOTIFY_COLORS['secondary'],
            gridwidth=0.5,
            showline=True,
            linecolor=SPOTIFY_COLORS['secondary']
        ),
        showlegend=True,
        legend=dict(
            bgcolor=SPOTIFY_COLORS['background'],
            bordercolor=SPOTIFY_COLORS['secondary'],
            borderwidth=1
        ),
        hovermode='x unified'
    )
    
    return fig

# Create top artists figure
def create_top_artists():
    top_artists = df.nlargest(10, 'Monthly Listeners (Millions)')
    
    fig = go.Figure(go.Bar(
        x=top_artists['Artist'],
        y=top_artists['Monthly Listeners (Millions)'],
        marker_color=SPOTIFY_COLORS['primary'],
        hovertemplate="<b>%{x}</b><br>" +
                     "Monthly Listeners: %{y:.1f}M<br>" +
                     "<extra></extra>"
    ))
    
    fig.update_layout(
        plot_bgcolor=SPOTIFY_COLORS['background'],
        paper_bgcolor=SPOTIFY_COLORS['background'],
        font_color=SPOTIFY_COLORS['text'],
        title={
            'text': 'Top 10 Artists by Monthly Listeners',
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'size': 24}
        },
        xaxis=dict(
            title='Artist',
            tickangle=45,
            gridcolor=SPOTIFY_COLORS['secondary'],
            showline=True,
            linecolor=SPOTIFY_COLORS['secondary']
        ),
        yaxis=dict(
            title='Monthly Listeners (Millions)',
            gridcolor=SPOTIFY_COLORS['secondary'],
            showline=True,
            linecolor=SPOTIFY_COLORS['secondary']
        ),
        showlegend=False,
        margin=dict(b=100)  # Add bottom margin for rotated labels
    )
    
    return fig

# Create world map figure
def create_world_map():
    fig = go.Figure(data=go.Choropleth(
        locations=df['Country'],
        locationmode='country names',
        z=df['Total Streams (Millions)'],
        colorscale=[[0, SPOTIFY_COLORS['secondary']], [1, SPOTIFY_COLORS['primary']]],
        marker_line_color=SPOTIFY_COLORS['text'],
        marker_line_width=0.5,
        colorbar_title="Streams (M)",
        hovertemplate="<b>%{location}</b><br>" +
                     "Total Streams: %{z:.0f}M<br>" +
                     "<extra></extra>"
    ))
    
    fig.update_layout(
        plot_bgcolor=SPOTIFY_COLORS['background'],
        paper_bgcolor=SPOTIFY_COLORS['background'],
        font_color=SPOTIFY_COLORS['text'],
        title={
            'text': 'Global Streaming Distribution',
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'size': 24}
        },
        geo=dict(
            showframe=False,
            showcoastlines=True,
            projection_type='equirectangular',
            bgcolor=SPOTIFY_COLORS['background'],
            coastlinecolor=SPOTIFY_COLORS['secondary'],
            showland=True,
            landcolor=SPOTIFY_COLORS['background'],
            showocean=True,
            oceancolor=SPOTIFY_COLORS['background']
        ),
        margin=dict(l=0, r=0, t=50, b=0)
    )
    
    return fig

# Layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("Spotify Global Streaming Analytics Dashboard", 
                       className="text-center my-4",
                       style={'color': SPOTIFY_COLORS['primary']}), width=12)
    ]),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Global Streaming Trends", style={'color': SPOTIFY_COLORS['primary']}),
                    dcc.Graph(figure=create_streaming_trends())
                ])
            ], style={'backgroundColor': SPOTIFY_COLORS['background'], 'border': f'1px solid {SPOTIFY_COLORS["secondary"]}'})
        ], width=12)
    ]),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Top Artists", style={'color': SPOTIFY_COLORS['primary']}),
                    dcc.Graph(figure=create_top_artists())
                ])
            ], style={'backgroundColor': SPOTIFY_COLORS['background'], 'border': f'1px solid {SPOTIFY_COLORS["secondary"]}'})
        ], width=6),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Geographic Distribution", style={'color': SPOTIFY_COLORS['primary']}),
                    dcc.Graph(figure=create_world_map())
                ])
            ], style={'backgroundColor': SPOTIFY_COLORS['background'], 'border': f'1px solid {SPOTIFY_COLORS["secondary"]}'})
        ], width=6)
    ], className="mt-4"),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Stream Predictor", 
                           style={
                               'color': SPOTIFY_COLORS['primary'],
                               'fontSize': '24px',
                               'textAlign': 'center',
                               'marginBottom': '20px'
                           }),
                    html.P("Predict how many streams a song might get based on different factors:", 
                          style={
                              'color': SPOTIFY_COLORS['text'],
                              'fontSize': '16px',
                              'textAlign': 'center',
                              'marginBottom': '30px'
                          }),
                    
                    dbc.Row([
                        dbc.Col([
                            # Left column - Dropdowns
                            html.Div([
                                dbc.Label("1. Select Country", 
                                         style={
                                             'color': SPOTIFY_COLORS['primary'],
                                             'fontWeight': 'bold',
                                             'fontSize': '16px',
                                             'marginBottom': '5px'
                                         }),
                                dcc.Dropdown(
                                    id='country-dropdown',
                                    options=[{'label': x, 'value': x} for x in sorted(df['Country'].unique())],
                                    value=df['Country'].iloc[0],
                                    clearable=False,
                                    style={
                                        'color': 'black',
                                        'backgroundColor': 'white',
                                        'marginBottom': '20px'
                                    }
                                ),
                                
                                dbc.Label("2. Choose Artist", 
                                         style={
                                             'color': SPOTIFY_COLORS['primary'],
                                             'fontWeight': 'bold',
                                             'fontSize': '16px',
                                             'marginBottom': '5px'
                                         }),
                                dcc.Dropdown(
                                    id='artist-dropdown',
                                    options=[{'label': x, 'value': x} for x in sorted(df['Artist'].unique())],
                                    value=df['Artist'].iloc[0],
                                    clearable=False,
                                    style={
                                        'color': 'black',
                                        'backgroundColor': 'white',
                                        'marginBottom': '20px'
                                    }
                                ),
                                
                                dbc.Label("3. Select Genre", 
                                         style={
                                             'color': SPOTIFY_COLORS['primary'],
                                             'fontWeight': 'bold',
                                             'fontSize': '16px',
                                             'marginBottom': '5px'
                                         }),
                                dcc.Dropdown(
                                    id='genre-dropdown',
                                    options=[{'label': x, 'value': x} for x in sorted(df['Genre'].unique())],
                                    value=df['Genre'].iloc[0],
                                    clearable=False,
                                    style={
                                        'color': 'black',
                                        'backgroundColor': 'white',
                                        'marginBottom': '20px'
                                    }
                                ),
                            ], style={'padding': '10px'})
                        ], width=6),
                        
                        dbc.Col([
                            # Right column - Numeric inputs
                            html.Div([
                                dbc.Label("4. Release Year (2018-2024)", 
                                         style={
                                             'color': SPOTIFY_COLORS['primary'],
                                             'fontWeight': 'bold',
                                             'fontSize': '16px'
                                         }),
                                dbc.Input(
                                    id='release-year',
                                    type='number',
                                    min=2018,
                                    max=2024,
                                    value=2024,
                                    style={
                                        'backgroundColor': 'white',
                                        'color': 'black',
                                        'marginBottom': '20px',
                                        'fontSize': '16px'
                                    }
                                ),
                                
                                dbc.Label("5. Monthly Listeners (Millions)", 
                                         style={
                                             'color': SPOTIFY_COLORS['primary'],
                                             'fontWeight': 'bold',
                                             'fontSize': '16px'
                                         }),
                                dbc.Input(
                                    id='monthly-listeners',
                                    type='number',
                                    min=0,
                                    step=0.1,
                                    value=1,
                                    style={
                                        'backgroundColor': 'white',
                                        'color': 'black',
                                        'marginBottom': '20px',
                                        'fontSize': '16px'
                                    }
                                ),
                                
                                dbc.Label("6. Average Song Duration (Minutes)", 
                                         style={
                                             'color': SPOTIFY_COLORS['primary'],
                                             'fontWeight': 'bold',
                                             'fontSize': '16px'
                                         }),
                                dbc.Input(
                                    id='stream-duration',
                                    type='number',
                                    min=0,
                                    step=0.5,
                                    value=3,
                                    style={
                                        'backgroundColor': 'white',
                                        'color': 'black',
                                        'marginBottom': '20px',
                                        'fontSize': '16px'
                                    }
                                ),
                                
                                dbc.Label("7. Skip Rate (0-100%)", 
                                         style={
                                             'color': SPOTIFY_COLORS['primary'],
                                             'fontWeight': 'bold',
                                             'fontSize': '16px'
                                         }),
                                dbc.Input(
                                    id='skip-rate',
                                    type='number',
                                    min=0,
                                    max=100,
                                    value=20,
                                    style={
                                        'backgroundColor': 'white',
                                        'color': 'black',
                                        'marginBottom': '20px',
                                        'fontSize': '16px'
                                    }
                                ),
                            ], style={'padding': '10px'})
                        ], width=6)
                    ]),
                    
                    # Predict Button
                    dbc.Row([
                        dbc.Col([
                            dbc.Button(
                                "Calculate Streams",
                                id='predict-button',
                                color='success',
                                size='lg',
                                className='w-100 mb-3',
                                style={
                                    'fontSize': '20px',
                                    'fontWeight': 'bold',
                                    'padding': '15px',
                                    'backgroundColor': SPOTIFY_COLORS['primary'],
                                    'borderColor': SPOTIFY_COLORS['primary']
                                }
                            ),
                        ], width=8, className='mx-auto')
                    ]),
                    
                    # Results Display
                    dbc.Alert(
                        id='prediction-output',
                        color='success',
                        is_open=False,
                        style={
                            'width': '100%',
                            'textAlign': 'center',
                            'marginTop': '20px',
                            'fontSize': '18px'
                        }
                    )
                ])
            ], style={
                'backgroundColor': SPOTIFY_COLORS['background'],
                'border': f'1px solid {SPOTIFY_COLORS["secondary"]}',
                'borderRadius': '15px',
                'boxShadow': '0 4px 8px rgba(0, 0, 0, 0.2)'
            })
        ], width=12)
    ], className="mt-4 mb-4")
], fluid=True, style={'backgroundColor': SPOTIFY_COLORS['background'], 'minHeight': '100vh'})

@app.callback(
    [Output('prediction-output', 'children'),
     Output('prediction-output', 'is_open'),
     Output('prediction-output', 'color')],
    [Input('predict-button', 'n_clicks')],
    [State('country-dropdown', 'value'),
     State('artist-dropdown', 'value'),
     State('genre-dropdown', 'value'),
     State('release-year', 'value'),
     State('monthly-listeners', 'value'),
     State('stream-duration', 'value'),
     State('skip-rate', 'value')]
)
def update_prediction(n_clicks, country, artist, genre, release_year, monthly_listeners, 
                     stream_duration, skip_rate):
    if n_clicks == 0:
        return "", False, "success"
    
    if None in [country, artist, genre, release_year, monthly_listeners, stream_duration, skip_rate]:
        return "Please fill in all fields", True, "warning"
    
    try:
        # Validate numeric inputs
        if not (2018 <= release_year <= 2024):
            return "Release year must be between 2018 and 2024", True, "danger"
        
        if monthly_listeners < 0:
            return "Monthly listeners must be positive", True, "danger"
        
        if stream_duration <= 0:
            return "Stream duration must be positive", True, "danger"
        
        if not (0 <= skip_rate <= 100):
            return "Skip rate must be between 0 and 100", True, "danger"
        
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
        
        return [
            html.Div([
                html.H4("Predicted Total Streams", className="mb-2"),
                html.H2(f"{prediction:.1f}M", style={'color': SPOTIFY_COLORS['primary']})
            ]),
            True,
            "success"
        ]
    
    except Exception as e:
        print(f"Error in prediction: {str(e)}")  # For debugging
        return f"An error occurred: {str(e)}", True, "danger"

if __name__ == '__main__':
    app.run_server(debug=True) 