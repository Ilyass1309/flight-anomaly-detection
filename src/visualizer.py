"""
Visualization Functions
Create professional plots for flight data analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List, Tuple


# Set professional style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def plot_flight_analysis(
    df: pd.DataFrame,
    save_path: Optional[str] = None,
    show_anomalies: bool = True,
    figsize: Tuple[int, int] = (15, 10)
) -> plt.Figure:
    """
    Create comprehensive flight analysis visualization.
    
    Parameters
    ----------
    df : pd.DataFrame
        Flight data with detection results
    save_path : str, optional
        Path to save the figure
    show_anomalies : bool, default=True
        Whether to highlight anomalies
    figsize : tuple, default=(15, 10)
        Figure size (width, height)
        
    Returns
    -------
    matplotlib.figure.Figure
        The created figure
    """
    fig, axes = plt.subplots(4, 1, figsize=figsize)
    fig.suptitle('Flight Data Analysis with Anomaly Detection', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    # Plot 1: Altitude
    axes[0].plot(df['timestamp'], df['altitude_ft'], 
                 label='Altitude', alpha=0.7, linewidth=1)
    
    if show_anomalies and 'altitude_ft_anomaly' in df.columns:
        anomalies = df[df['altitude_ft_anomaly']]
        axes[0].scatter(anomalies['timestamp'], anomalies['altitude_ft'],
                       color='red', s=50, label='Anomalies', zorder=5, alpha=0.8)
    
    axes[0].set_ylabel('Altitude (ft)', fontsize=11, fontweight='bold')
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title('Altitude Over Time', fontsize=12, pad=10)
    
    # Plot 2: Speed
    axes[1].plot(df['timestamp'], df['speed_knots'],
                 label='Speed', alpha=0.7, linewidth=1, color='orange')
    
    if show_anomalies and 'speed_knots_anomaly' in df.columns:
        anomalies = df[df['speed_knots_anomaly']]
        axes[1].scatter(anomalies['timestamp'], anomalies['speed_knots'],
                       color='red', s=50, label='Anomalies', zorder=5, alpha=0.8)
    
    axes[1].set_ylabel('Speed (knots)', fontsize=11, fontweight='bold')
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_title('Speed Over Time', fontsize=12, pad=10)
    
    # Plot 3: Fuel
    axes[2].plot(df['timestamp'], df['fuel_lbs'],
                 label='Fuel', alpha=0.7, linewidth=1, color='green')
    axes[2].set_ylabel('Fuel (lbs)', fontsize=11, fontweight='bold')
    axes[2].legend(loc='upper right')
    axes[2].grid(True, alpha=0.3)
    axes[2].set_title('Fuel Consumption Over Time', fontsize=12, pad=10)
    
    # Plot 4: Engine Temperature
    axes[3].plot(df['timestamp'], df['engine_temp_c'],
                 label='Engine Temp', alpha=0.7, linewidth=1, color='purple')
    
    if show_anomalies and 'engine_temp_c_anomaly' in df.columns:
        anomalies = df[df['engine_temp_c_anomaly']]
        axes[3].scatter(anomalies['timestamp'], anomalies['engine_temp_c'],
                       color='red', s=50, label='Anomalies', zorder=5, alpha=0.8)
    
    axes[3].set_ylabel('Temperature (°C)', fontsize=11, fontweight='bold')
    axes[3].set_xlabel('Time', fontsize=11, fontweight='bold')
    axes[3].legend(loc='upper right')
    axes[3].grid(True, alpha=0.3)
    axes[3].set_title('Engine Temperature Over Time', fontsize=12, pad=10)
    
    # Rotate x-axis labels for better readability
    for ax in axes:
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    return fig


def plot_anomaly_distribution(
    df: pd.DataFrame,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 6)
) -> plt.Figure:
    """
    Plot distribution of anomalies across parameters.
    
    Parameters
    ----------
    df : pd.DataFrame
        Flight data with anomaly detection results
    save_path : str, optional
        Path to save the figure
    figsize : tuple, default=(12, 6)
        Figure size
        
    Returns
    -------
    matplotlib.figure.Figure
        The created figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle('Anomaly Distribution Analysis', fontsize=14, fontweight='bold')
    
    # Count anomalies per parameter
    anomaly_counts = {}
    for col in df.columns:
        if col.endswith('_anomaly') and col != 'anomaly':
            param_name = col.replace('_anomaly', '')
            anomaly_counts[param_name] = df[col].sum()
    
    # Plot 1: Bar chart of anomaly counts
    params = list(anomaly_counts.keys())
    counts = list(anomaly_counts.values())
    
    axes[0].bar(params, counts, color='coral', alpha=0.7, edgecolor='black')
    axes[0].set_ylabel('Number of Anomalies', fontweight='bold')
    axes[0].set_title('Anomalies by Parameter', fontsize=12)
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (param, count) in enumerate(zip(params, counts)):
        axes[0].text(i, count, str(count), ha='center', va='bottom', fontweight='bold')
    
    # Plot 2: Pie chart of anomaly distribution
    if 'anomaly' in df.columns:
        normal_count = (~df['anomaly']).sum()
        anomaly_count = df['anomaly'].sum()
        
        labels = ['Normal', 'Anomaly']
        sizes = [normal_count, anomaly_count]
        colors = ['lightgreen', 'lightcoral']
        explode = (0, 0.1)
        
        axes[1].pie(sizes, explode=explode, labels=labels, colors=colors,
                   autopct='%1.1f%%', shadow=True, startangle=90)
        axes[1].set_title('Overall Data Distribution', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    return fig


def plot_correlation_matrix(
    df: pd.DataFrame,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8)
) -> plt.Figure:
    """
    Plot correlation matrix of flight parameters.
    
    Parameters
    ----------
    df : pd.DataFrame
        Flight data
    save_path : str, optional
        Path to save the figure
    figsize : tuple, default=(10, 8)
        Figure size
        
    Returns
    -------
    matplotlib.figure.Figure
        The created figure
    """
    # Select only numeric columns (exclude timestamp and anomaly flags)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if not col.endswith('_anomaly') 
                    and not col.endswith('_zscore') and col != 'anomaly']
    
    # Calculate correlation matrix
    corr_matrix = df[numeric_cols].corr()
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8},
                ax=ax)
    
    ax.set_title('Correlation Matrix of Flight Parameters', 
                 fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    return fig


if __name__ == "__main__":
    print("This module contains visualization functions.")
    print("To test it, run: python test_visualizer.py")
