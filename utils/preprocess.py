import pandas as pd

def preprocess_data(csv_path):
    df = pd.read_csv(csv_path)

    # Manual encoding (CONSISTENT & MEANINGFUL)
    weather_map = {"Clear": 0, "Rain": 1, "Fog": 2}
    visibility_map = {"High": 0, "Medium": 1, "Low": 2}
    road_map = {"Urban": 0, "Highway": 1}
    severity_map = {"Low": 0, "Medium": 1, "High": 2}

    df['weather'] = df['weather'].map(weather_map)
    df['visibility'] = df['visibility'].map(visibility_map)
    df['road_type'] = df['road_type'].map(road_map)
    df['severity'] = df['severity'].map(severity_map)

    X = df[['weather', 'visibility', 'vehicles_involved', 'road_type']]
    y = df['severity']

    return X, y
