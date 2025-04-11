# Spotify Global Streaming Dashboard

An interactive dashboard that visualizes and analyzes Spotify's global streaming data, featuring machine learning predictions for stream counts.

## Features

1. **Global Streaming Trends**
   - Interactive line chart showing streaming trends by genre over time
   - Color-coded genre visualization
   - Hover tooltips with detailed information

2. **Top Artists Analysis**
   - Bar chart of top 10 artists by monthly listeners
   - Interactive tooltips with statistics
   - Clean and modern visualization

3. **Geographic Distribution**
   - World map showing streaming distribution
   - Color gradient based on stream counts
   - Country-specific hover information

4. **ML Stream Predictor**
   - Predict potential streams based on multiple factors:
     - Country
     - Artist
     - Genre
     - Release Year
     - Monthly Listeners
     - Average Song Duration
     - Skip Rate

## Technology Stack

- **Frontend**: Dash with Bootstrap components
- **Data Processing**: Pandas, NumPy
- **Visualization**: Plotly
- **Machine Learning**: Scikit-learn (RandomForestRegressor)
- **Deployment**: Streamlit Cloud

## Local Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/hosseinmang/Spotify_Global_Streaming_Dashboard.git
   cd Spotify_Global_Streaming_Dashboard
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the application:
   ```bash
   python app.py
   ```

5. Open your browser and navigate to:
   ```
   http://127.0.0.1:8050
   ```

## Data Sources

The dashboard uses the Spotify 2024 Global Streaming Data dataset, which includes information about:
- Artist statistics
- Song performance metrics
- Geographic distribution
- Genre-specific trends

## Contributing

Feel free to open issues or submit pull requests for any improvements you'd like to suggest.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 