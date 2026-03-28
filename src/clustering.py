"""
Clustering module for Uber-Pickups spatial analysis.
Contains implementations for KMeans and DBSCAN algorithms.
"""
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN

def apply_kmeans(df: pd.DataFrame, n_clusters: int = 8, random_state: int = 42) -> pd.DataFrame:
    """
    Apply KMeans clustering to geographical coordinates.
    
    Parameters:
    - df (pd.DataFrame): Dataset containing 'Lat' and 'Lon' columns.
    - n_clusters (int): The target number of centroids.
    - random_state (int): Seed for reproducibility.
    
    Returns:
    - pd.DataFrame: The dataset updated with a 'Cluster_KMeans' string column.
    """
    X = df[['Lat', 'Lon']]
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init='auto')
    
    df_result = df.copy()
    df_result['Cluster_KMeans'] = kmeans.fit_predict(X).astype(str)
    
    return df_result

def apply_dbscan(df: pd.DataFrame, eps: float = 0.005, min_samples: int = 15) -> pd.DataFrame:
    """
    Apply DBSCAN density-based clustering to geographical coordinates.
    
    Parameters:
    - df (pd.DataFrame): Dataset containing 'Lat' and 'Lon' columns.
    - eps (float): The maximum distance between two samples for one to be considered as in the neighborhood of the other.
    - min_samples (int): The number of samples in a neighborhood for a point to be considered as a core point.
    
    Returns:
    - pd.DataFrame: The dataset updated with a 'Cluster_DBSCAN' string column.
    """
    X = df[['Lat', 'Lon']]
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    
    df_result = df.copy()
    df_result['Cluster_DBSCAN'] = dbscan.fit_predict(X).astype(str)
    
    return df_result