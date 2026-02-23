"""
Flight Anomaly Detection Dashboard
Interactive application for flight telemetry anomaly detection and analysis.
"""

import streamlit as st
from src.data_generator import generate_flight_data
from src.detector import AnomalyDetector
from src.ml_detector import IsolationForestDetector
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
st.set_page_config(
    page_title="Flight Anomaly Detection System",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        margin-top: 2rem;
        margin-bottom: 1rem;
        color: #2c3e50;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<p class="main-header">Flight Anomaly Detection System</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Advanced Statistical and Machine Learning Analysis Platform</p>', unsafe_allow_html=True)
st.divider()

# Sidebar Configuration
st.sidebar.title("Configuration Panel")

# Data Generation Settings
st.sidebar.header("1. Data Parameters")
n_samples = st.sidebar.number_input(
    "Sample Size",
    min_value=100,
    max_value=100000,
    value=10000,
    step=1000,
    help="Number of data points to generate"
)

anomaly_rate = st.sidebar.slider(
    "Injected Anomaly Rate",
    min_value=0.01,
    max_value=0.25,
    value=0.05,
    step=0.01,
    format="%.2f",
    help="Proportion of anomalies to inject into the dataset"
)

st.sidebar.divider()

# Detection Method Selection
st.sidebar.header("2. Detection Method")
method = st.sidebar.selectbox(
    "Algorithm",
    ["Statistical (Z-Score)", "Machine Learning (Isolation Forest)"],
    help="Select the anomaly detection algorithm"
)

# Method-specific Parameters
st.sidebar.header("3. Algorithm Parameters")

# Initialize all parameters with default values
threshold = 3.0
contamination = 0.05
n_estimators = 100

if method == "Statistical (Z-Score)":
    threshold = st.sidebar.slider(
        "Sigma Threshold",
        min_value=1.0,
        max_value=5.0,
        value=3.0,
        step=0.1,
        help="Number of standard deviations for anomaly classification"
    )
    st.sidebar.caption("Standard: 3σ captures 99.7% of normal distribution")
    
    # Also show ML parameters for comparison mode
    with st.sidebar.expander("Isolation Forest Parameters (for comparison)", expanded=False):
        contamination = st.slider(
            "Contamination Factor",
            min_value=0.01,
            max_value=0.25,
            value=0.05,
            step=0.01,
            format="%.2f",
            key="contamination_comparison"
        )
        
        n_estimators = st.slider(
            "Number of Estimators",
            min_value=50,
            max_value=500,
            value=100,
            step=50,
            key="n_estimators_comparison"
        )
else:
    contamination = st.sidebar.slider(
        "Contamination Factor",
        min_value=0.01,
        max_value=0.25,
        value=0.05,
        step=0.01,
        format="%.2f",
        help="Expected proportion of outliers in the dataset"
    )
    
    n_estimators = st.sidebar.slider(
        "Number of Estimators",
        min_value=50,
        max_value=500,
        value=100,
        step=50,
        help="Number of trees in the isolation forest"
    )
    
    st.sidebar.caption("Higher values improve accuracy but increase computation time")
    
    # Also show Z-Score parameters for comparison mode
    with st.sidebar.expander("Z-Score Parameters (for comparison)", expanded=False):
        threshold = st.slider(
            "Sigma Threshold",
            min_value=1.0,
            max_value=5.0,
            value=3.0,
            step=0.1,
            key="threshold_comparison"
        )

st.sidebar.divider()

# Execution Buttons
col_btn1, col_btn2 = st.sidebar.columns(2)

with col_btn1:
    execute_analysis = st.sidebar.button(
        "Execute Analysis",
        type="primary",
        use_container_width=True
    )

with col_btn2:
    compare_methods = st.sidebar.button(
        "Compare Methods",
        type="secondary",
        use_container_width=True
    )

# Visualization options (always enabled)
show_scores = True
show_distribution = True

# Main Content Area
if execute_analysis:
    
    # Progress tracking
    status_container = st.empty()
    progress_bar = st.progress(0)
    
    # Step 1: Data Generation
    status_container.info("Status: Generating synthetic flight telemetry data...")
    progress_bar.progress(25)
    
    df = generate_flight_data(
        n_samples=n_samples,
        anomaly_rate=anomaly_rate,
        seed=42
    )
    
    # Step 2: Anomaly Detection
    status_container.info("Status: Executing anomaly detection algorithm...")
    progress_bar.progress(50)
    
    if method == "Statistical (Z-Score)":
        detector = AnomalyDetector(method='zscore', threshold=threshold)
        results = detector.fit_detect(
            df,
            columns=['altitude_ft', 'speed_knots', 'engine_temp_c']
        )
        n_anomalies = results['anomaly'].sum()
        anomaly_column = 'anomaly'
        method_short = "Z-Score"
    else:
        detector = IsolationForestDetector(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=42
        )
        results = detector.fit_detect(
            df,
            columns=['altitude_ft', 'speed_knots', 'engine_temp_c']
        )
        n_anomalies = results['anomaly_ml'].sum()
        anomaly_column = 'anomaly_ml'
        method_short = "Isolation Forest"
    
    # Step 3: Analysis Completion
    status_container.info("Status: Generating analysis results...")
    progress_bar.progress(75)
    
    # Finalize
    progress_bar.progress(100)
    status_container.empty()
    progress_bar.empty()
    
    # Success notification
    st.success(f"Analysis completed successfully using {method_short} method")
    
    # Key Performance Indicators
    st.markdown("### Executive Summary")
    
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    
    with kpi1:
        st.metric(
            label="Total Data Points",
            value=f"{len(df):,}"
        )
    
    with kpi2:
        st.metric(
            label="Anomalies Detected",
            value=f"{n_anomalies:,}",
            delta=None
        )
    
    with kpi3:
        detection_rate = (n_anomalies / len(df)) * 100
        st.metric(
            label="Detection Rate",
            value=f"{detection_rate:.2f}%"
        )
    
    with kpi4:
        expected_rate = anomaly_rate * 100
        difference = detection_rate - expected_rate
        delta_color = "normal" if abs(difference) < 1 else "inverse"
        st.metric(
            label="Expected Rate",
            value=f"{expected_rate:.2f}%",
            delta=f"{difference:+.2f}%"
        )
    
    st.divider()
    
    # Detailed Analysis Section
    st.markdown("### Detailed Analysis")
    
    tab1, tab2, tab3 = st.tabs(["Statistical Summary", "Data Preview", "Visualizations"])
    
    with tab1:
        if method == "Statistical (Z-Score)":
            st.markdown("#### Parameter-wise Anomaly Distribution")
            summary = detector.get_anomaly_summary(results)
            st.dataframe(
                summary,
                use_container_width=True,
                hide_index=True
            )
        else:
            st.markdown("#### Anomaly Score Distribution Statistics")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Overall Score Metrics**")
                score_stats = {
                    "Metric": ["Minimum", "Maximum", "Mean", "Median", "Std. Deviation"],
                    "Value": [
                        f"{results['anomaly_score'].min():.4f}",
                        f"{results['anomaly_score'].max():.4f}",
                        f"{results['anomaly_score'].mean():.4f}",
                        f"{results['anomaly_score'].median():.4f}",
                        f"{results['anomaly_score'].std():.4f}"
                    ]
                }
                st.dataframe(pd.DataFrame(score_stats), hide_index=True, use_container_width=True)
            
            with col2:
                st.markdown("**Comparative Analysis**")
                normal_scores = results[~results[anomaly_column]]['anomaly_score']
                anomaly_scores = results[results[anomaly_column]]['anomaly_score']
                
                comparison_stats = {
                    "Category": ["Normal Points", "Anomalous Points", "Separation Index"],
                    "Mean Score": [
                        f"{normal_scores.mean():.4f}",
                        f"{anomaly_scores.mean():.4f}",
                        f"{abs(normal_scores.mean() - anomaly_scores.mean()):.4f}"
                    ]
                }
                st.dataframe(pd.DataFrame(comparison_stats), hide_index=True, use_container_width=True)
    
    with tab2:
        st.markdown("#### Anomalous Observations (Top 15)")
        
        anomalies_sample = results[results[anomaly_column]].head(15)
        
        if len(anomalies_sample) > 0:
            display_cols = ['timestamp', 'altitude_ft', 'speed_knots', 'engine_temp_c', 'fuel_lbs']
            if method_short == "Isolation Forest" and show_scores:
                display_cols.append('anomaly_score')
            
            st.dataframe(
                anomalies_sample[display_cols].reset_index(drop=True),
                use_container_width=True
            )
        else:
            st.info("No anomalies detected with current configuration")
    
    with tab3:
        if method_short == "Isolation Forest":
            st.markdown("#### Score Distribution Analysis")
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            
            # Histogram
            ax1.hist(results['anomaly_score'], bins=50, alpha=0.7, color='steelblue', edgecolor='black')
            ax1.axvline(results['anomaly_score'].mean(), color='red', linestyle='--', linewidth=2, label='Mean')
            ax1.set_xlabel('Anomaly Score', fontweight='bold')
            ax1.set_ylabel('Frequency', fontweight='bold')
            ax1.set_title('Score Distribution', fontweight='bold')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Box plot comparison
            box_data = [
                normal_scores.values,
                anomaly_scores.values
            ]
            bp = ax2.boxplot(box_data, labels=['Normal', 'Anomalous'], patch_artist=True)
            bp['boxes'][0].set_facecolor('lightgreen')
            bp['boxes'][1].set_facecolor('lightcoral')
            ax2.set_ylabel('Anomaly Score', fontweight='bold')
            ax2.set_title('Score Comparison by Category', fontweight='bold')
            ax2.grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.markdown("#### Parameter Distribution Analysis")
            
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            fig.suptitle('Statistical Analysis - Z-Score Method', fontweight='bold', fontsize=14)
            
            parameters = ['altitude_ft', 'speed_knots', 'engine_temp_c']
            param_labels = ['Altitude (ft)', 'Speed (knots)', 'Engine Temperature (°C)']
            
            # Plot distributions for each parameter
            for idx, (param, label) in enumerate(zip(parameters, param_labels)):
                row = idx // 2
                col = idx % 2
                ax = axes[row, col]
                
                normal_data = results[~results[anomaly_column]][param]
                anomaly_data = results[results[anomaly_column]][param]
                
                ax.hist(normal_data, bins=40, alpha=0.6, color='lightgreen', label='Normal', edgecolor='black')
                ax.hist(anomaly_data, bins=40, alpha=0.6, color='lightcoral', label='Anomalous', edgecolor='black')
                ax.set_xlabel(label, fontweight='bold')
                ax.set_ylabel('Frequency', fontweight='bold')
                ax.set_title(f'{label} Distribution', fontweight='bold')
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            # Anomaly count by parameter
            ax = axes[1, 1]
            anomaly_counts = []
            for param in parameters:
                z_col = f'{param}_zscore'
                if z_col in results.columns:
                    count = (abs(results[z_col]) > threshold).sum()
                    anomaly_counts.append(count)
                else:
                    anomaly_counts.append(0)
            
            bars = ax.bar(param_labels, anomaly_counts, color=['steelblue', 'orange', 'crimson'], edgecolor='black')
            ax.set_ylabel('Anomaly Count', fontweight='bold')
            ax.set_title('Anomalies by Parameter', fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=15, ha='right')
            
            plt.tight_layout()
            st.pyplot(fig)
    
    st.divider()
    
    # Export Section
    st.markdown("### Data Export")
    
    export_col1, export_col2, export_col3 = st.columns(3)
    
    with export_col1:
        csv_all = results.to_csv(index=False)
        st.download_button(
            label="Export Complete Dataset",
            data=csv_all,
            file_name=f"flight_anomaly_full_{method_short.lower().replace(' ', '_')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with export_col2:
        csv_anomalies = results[results[anomaly_column]].to_csv(index=False)
        st.download_button(
            label="Export Anomalies Only",
            data=csv_anomalies,
            file_name=f"anomalies_{method_short.lower().replace(' ', '_')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with export_col3:
        if method == "Statistical (Z-Score)":
            summary_csv = summary.to_csv(index=False)
        else:
            summary_df = pd.DataFrame({
                'Metric': ['Total Points', 'Anomalies', 'Detection Rate', 'Expected Rate'],
                'Value': [len(df), n_anomalies, f"{detection_rate:.2f}%", f"{expected_rate:.2f}%"]
            })
            summary_csv = summary_df.to_csv(index=False)
        
        st.download_button(
            label="Export Summary Report",
            data=summary_csv,
            file_name=f"summary_{method_short.lower().replace(' ', '_')}.csv",
            mime="text/csv",
            use_container_width=True
        )

elif compare_methods:
    # Method Comparison Mode
    st.info("Comparing Statistical (Z-Score) vs Machine Learning (Isolation Forest) methods...")
    
    # Progress tracking
    status_container = st.empty()
    progress_bar = st.progress(0)
    
    # Step 1: Data Generation
    status_container.info("Status: Generating synthetic flight telemetry data...")
    progress_bar.progress(20)
    
    df = generate_flight_data(
        n_samples=n_samples,
        anomaly_rate=anomaly_rate,
        seed=42
    )
    
    # Step 2: Statistical Detection
    status_container.info("Status: Running Z-Score detection...")
    progress_bar.progress(40)
    
    stat_detector = AnomalyDetector(method='zscore', threshold=threshold)
    stat_results = stat_detector.fit_detect(
        df,
        columns=['altitude_ft', 'speed_knots', 'engine_temp_c']
    )
    stat_anomalies = stat_results['anomaly'].sum()
    
    # Step 3: ML Detection
    status_container.info("Status: Running Isolation Forest detection...")
    progress_bar.progress(70)
    
    ml_detector = IsolationForestDetector(
        contamination=contamination,
        n_estimators=n_estimators,
        random_state=42
    )
    ml_results = ml_detector.fit_detect(
        df,
        columns=['altitude_ft', 'speed_knots', 'engine_temp_c']
    )
    ml_anomalies = ml_results['anomaly_ml'].sum()
    
    # Finalize
    progress_bar.progress(100)
    status_container.empty()
    progress_bar.empty()
    
    # Success notification
    st.success("Comparison analysis completed successfully!")
    
    # Comparison Summary
    st.markdown("### Comparison Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Data Points",
            value=f"{len(df):,}"
        )
    
    with col2:
        st.metric(
            label="Z-Score Detected",
            value=f"{stat_anomalies:,}",
            delta=f"{(stat_anomalies/len(df)*100):.2f}%"
        )
    
    with col3:
        st.metric(
            label="Isolation Forest Detected",
            value=f"{ml_anomalies:,}",
            delta=f"{(ml_anomalies/len(df)*100):.2f}%"
        )
    
    with col4:
        # Calculate agreement
        both = (stat_results['anomaly']) & (ml_results['anomaly_ml'])
        agreement = both.sum()
        agreement_rate = (agreement / max(stat_anomalies, ml_anomalies, 1)) * 100
        st.metric(
            label="Agreement",
            value=f"{agreement:,}",
            delta=f"{agreement_rate:.1f}%"
        )
    
    st.divider()
    
    # Detailed Comparison
    st.markdown("### Detailed Comparison Analysis")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Detection Overlap", "Visual Comparison", "Performance Metrics", "Data Export"])
    
    with tab1:
        st.markdown("#### Detection Agreement Analysis")
        
        col_a, col_b = st.columns(2)
        
        with col_a:
            # Venn diagram data
            both_detected = (stat_results['anomaly']) & (ml_results['anomaly_ml'])
            only_stat = (stat_results['anomaly']) & (~ml_results['anomaly_ml'])
            only_ml = (~stat_results['anomaly']) & (ml_results['anomaly_ml'])
            
            venn_data = {
                'Category': [
                    'Detected by Both Methods',
                    'Only Z-Score',
                    'Only Isolation Forest',
                    'Total Unique Anomalies'
                ],
                'Count': [
                    both_detected.sum(),
                    only_stat.sum(),
                    only_ml.sum(),
                    stat_anomalies + ml_anomalies - both_detected.sum()
                ],
                'Percentage': [
                    f"{(both_detected.sum()/len(df)*100):.2f}%",
                    f"{(only_stat.sum()/len(df)*100):.2f}%",
                    f"{(only_ml.sum()/len(df)*100):.2f}%",
                    f"{((stat_anomalies + ml_anomalies - both_detected.sum())/len(df)*100):.2f}%"
                ]
            }
            
            st.dataframe(
                pd.DataFrame(venn_data),
                hide_index=True,
                use_container_width=True
            )
        
        with col_b:
            # Pie chart of overlap
            fig_pie, ax_pie = plt.subplots(figsize=(8, 6))
            
            sizes = [both_detected.sum(), only_stat.sum(), only_ml.sum()]
            labels = ['Both Methods', 'Z-Score Only', 'Isolation Forest Only']
            colors = ['#66b3ff', '#ff9999', '#99ff99']
            explode = (0.1, 0, 0)
            
            ax_pie.pie(sizes, explode=explode, labels=labels, colors=colors,
                      autopct='%1.1f%%', shadow=True, startangle=90)
            ax_pie.set_title('Detection Method Overlap', fontweight='bold', fontsize=12)
            
            st.pyplot(fig_pie)
    
    with tab2:
        st.markdown("#### Visual Comparison: Altitude vs Speed")
        
        fig_comp, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Z-Score plot
        scatter1 = ax1.scatter(
            stat_results[~stat_results['anomaly']]['altitude_ft'],
            stat_results[~stat_results['anomaly']]['speed_knots'],
            c='lightblue', alpha=0.5, s=20, label='Normal', edgecolors='none'
        )
        scatter1_anom = ax1.scatter(
            stat_results[stat_results['anomaly']]['altitude_ft'],
            stat_results[stat_results['anomaly']]['speed_knots'],
            c='red', alpha=0.7, s=40, label='Anomaly', edgecolors='black', linewidth=0.5
        )
        ax1.set_xlabel('Altitude (ft)', fontweight='bold')
        ax1.set_ylabel('Speed (knots)', fontweight='bold')
        ax1.set_title(f'Z-Score Method\n({stat_anomalies} anomalies)', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Isolation Forest plot
        scatter2 = ax2.scatter(
            ml_results[~ml_results['anomaly_ml']]['altitude_ft'],
            ml_results[~ml_results['anomaly_ml']]['speed_knots'],
            c='lightgreen', alpha=0.5, s=20, label='Normal', edgecolors='none'
        )
        scatter2_anom = ax2.scatter(
            ml_results[ml_results['anomaly_ml']]['altitude_ft'],
            ml_results[ml_results['anomaly_ml']]['speed_knots'],
            c='darkred', alpha=0.7, s=40, label='Anomaly', edgecolors='black', linewidth=0.5
        )
        ax2.set_xlabel('Altitude (ft)', fontweight='bold')
        ax2.set_ylabel('Speed (knots)', fontweight='bold')
        ax2.set_title(f'Isolation Forest Method\n({ml_anomalies} anomalies)', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig_comp)
        
        # Engine Temperature comparison
        st.markdown("#### Visual Comparison: Altitude vs Engine Temperature")
        
        fig_comp2, (ax3, ax4) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Z-Score plot
        ax3.scatter(
            stat_results[~stat_results['anomaly']]['altitude_ft'],
            stat_results[~stat_results['anomaly']]['engine_temp_c'],
            c='lightblue', alpha=0.5, s=20, label='Normal', edgecolors='none'
        )
        ax3.scatter(
            stat_results[stat_results['anomaly']]['altitude_ft'],
            stat_results[stat_results['anomaly']]['engine_temp_c'],
            c='red', alpha=0.7, s=40, label='Anomaly', edgecolors='black', linewidth=0.5
        )
        ax3.set_xlabel('Altitude (ft)', fontweight='bold')
        ax3.set_ylabel('Engine Temperature (°C)', fontweight='bold')
        ax3.set_title('Z-Score Method', fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Isolation Forest plot
        ax4.scatter(
            ml_results[~ml_results['anomaly_ml']]['altitude_ft'],
            ml_results[~ml_results['anomaly_ml']]['engine_temp_c'],
            c='lightgreen', alpha=0.5, s=20, label='Normal', edgecolors='none'
        )
        ax4.scatter(
            ml_results[ml_results['anomaly_ml']]['altitude_ft'],
            ml_results[ml_results['anomaly_ml']]['engine_temp_c'],
            c='darkred', alpha=0.7, s=40, label='Anomaly', edgecolors='black', linewidth=0.5
        )
        ax4.set_xlabel('Altitude (ft)', fontweight='bold')
        ax4.set_ylabel('Engine Temperature (°C)', fontweight='bold')
        ax4.set_title('Isolation Forest Method', fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig_comp2)
    
    with tab3:
        st.markdown("#### Performance Metrics Comparison")
        
        col_m1, col_m2 = st.columns(2)
        
        with col_m1:
            st.markdown("**Z-Score Method**")
            stat_metrics = {
                'Metric': ['Total Anomalies', 'Detection Rate', 'Precision vs Expected', 'Method Complexity'],
                'Value': [
                    f"{stat_anomalies:,}",
                    f"{(stat_anomalies/len(df)*100):.2f}%",
                    f"{abs((stat_anomalies/len(df)) - anomaly_rate)*100:.2f}% difference",
                    'Low (O(n))'
                ]
            }
            st.dataframe(pd.DataFrame(stat_metrics), hide_index=True, use_container_width=True)
        
        with col_m2:
            st.markdown("**Isolation Forest Method**")
            ml_metrics = {
                'Metric': ['Total Anomalies', 'Detection Rate', 'Precision vs Expected', 'Method Complexity'],
                'Value': [
                    f"{ml_anomalies:,}",
                    f"{(ml_anomalies/len(df)*100):.2f}%",
                    f"{abs((ml_anomalies/len(df)) - anomaly_rate)*100:.2f}% difference",
                    f'Medium (O(n log n))'
                ]
            }
            st.dataframe(pd.DataFrame(ml_metrics), hide_index=True, use_container_width=True)
        
        st.markdown("---")
        st.markdown("**Recommendation Analysis**")
        
        # Calculate which method is closer to expected rate
        stat_diff = abs((stat_anomalies/len(df)) - anomaly_rate)
        ml_diff = abs((ml_anomalies/len(df)) - anomaly_rate)
        
        if stat_diff < ml_diff:
            st.info("✓ **Z-Score method** is closer to the expected anomaly rate for this dataset.")
        elif ml_diff < stat_diff:
            st.info("✓ **Isolation Forest method** is closer to the expected anomaly rate for this dataset.")
        else:
            st.info("✓ Both methods achieved similar accuracy relative to the expected anomaly rate.")
        
        if agreement_rate > 70:
            st.success(f"✓ High agreement rate ({agreement_rate:.1f}%) indicates both methods are detecting similar anomaly patterns.")
        elif agreement_rate > 40:
            st.warning(f"⚠ Moderate agreement rate ({agreement_rate:.1f}%) - methods are detecting different types of anomalies.")
        else:
            st.error(f"⚠ Low agreement rate ({agreement_rate:.1f}%) - significant divergence between methods.")
    
    with tab4:
        st.markdown("#### Export Comparison Results")
        
        export_c1, export_c2, export_c3 = st.columns(3)
        
        with export_c1:
            # Combined results
            combined_results = df.copy()
            combined_results['zscore_anomaly'] = stat_results['anomaly']
            combined_results['ml_anomaly'] = ml_results['anomaly_ml']
            combined_results['anomaly_score'] = ml_results['anomaly_score']
            combined_results['detected_by_both'] = both_detected
            
            csv_combined = combined_results.to_csv(index=False)
            st.download_button(
                label="Export Combined Results",
                data=csv_combined,
                file_name="comparison_combined_results.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with export_c2:
            # Agreement analysis
            agreement_df = pd.DataFrame({
                'Method': ['Both Methods', 'Z-Score Only', 'Isolation Forest Only', 'Neither'],
                'Count': [
                    both_detected.sum(),
                    only_stat.sum(),
                    only_ml.sum(),
                    len(df) - stat_anomalies - ml_anomalies + both_detected.sum()
                ],
                'Percentage': [
                    f"{(both_detected.sum()/len(df)*100):.2f}%",
                    f"{(only_stat.sum()/len(df)*100):.2f}%",
                    f"{(only_ml.sum()/len(df)*100):.2f}%",
                    f"{((len(df) - stat_anomalies - ml_anomalies + both_detected.sum())/len(df)*100):.2f}%"
                ]
            })
            
            csv_agreement = agreement_df.to_csv(index=False)
            st.download_button(
                label="Export Agreement Analysis",
                data=csv_agreement,
                file_name="comparison_agreement.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with export_c3:
            # Summary report
            summary_comparison = pd.DataFrame({
                'Metric': [
                    'Total Data Points',
                    'Expected Anomalies',
                    'Z-Score Detected',
                    'Isolation Forest Detected',
                    'Agreement Count',
                    'Agreement Rate'
                ],
                'Value': [
                    len(df),
                    f"{int(len(df) * anomaly_rate)} ({anomaly_rate*100:.1f}%)",
                    f"{stat_anomalies} ({stat_anomalies/len(df)*100:.2f}%)",
                    f"{ml_anomalies} ({ml_anomalies/len(df)*100:.2f}%)",
                    agreement,
                    f"{agreement_rate:.1f}%"
                ]
            })
            
            csv_summary = summary_comparison.to_csv(index=False)
            st.download_button(
                label="Export Summary Report",
                data=csv_summary,
                file_name="comparison_summary.csv",
                mime="text/csv",
                use_container_width=True
            )

else:
    # Welcome screen with instructions
    st.markdown("### System Overview")
    
    st.markdown("""
    This application provides advanced anomaly detection capabilities for flight telemetry data 
    using both statistical and machine learning approaches. The system generates synthetic flight 
    data with controlled anomaly injection for testing and validation purposes.
    """)
    
    st.markdown("### Getting Started")
    
    st.markdown("""
    **Configuration Steps:**
    
    1. **Configure Data Parameters** - Set the sample size and anomaly injection rate
    2. **Select Detection Method** - Choose between statistical or machine learning approach
    3. **Adjust Algorithm Parameters** - Fine-tune method-specific settings
    4. **Execute Analysis** - Run a single detection method OR **Compare Methods** - Run both methods simultaneously
    5. **Review Results** - Examine detailed statistics and anomalous observations
    6. **Export Data** - Download results in CSV format
    
    **Two Analysis Modes:**
    - **Execute Analysis**: Run the selected method (Z-Score or Isolation Forest)
    - **Compare Methods**: Run both methods simultaneously and analyze their differences
    """)
    
    st.divider()
    
    st.markdown("### Detection Methodologies")
    
    method_col1, method_col2 = st.columns(2)
    
    with method_col1:
        st.markdown("#### Statistical Approach (Z-Score)")
        st.markdown("""
        **Characteristics:**
        - Computational efficiency: O(n)
        - Assumption: Normal distribution
        - Detection basis: Standard deviation
        - Best for: Univariate outliers
        
        **Use Cases:**
        - Quick screening
        - Real-time applications
        - Single-parameter analysis
        """)
    
    with method_col2:
        st.markdown("#### Machine Learning (Isolation Forest)")
        st.markdown("""
        **Characteristics:**
        - Computational complexity: O(n log n)
        - Assumption: None required
        - Detection basis: Isolation difficulty
        - Best for: Multivariate patterns
        
        **Use Cases:**
        - Complex anomaly patterns
        - Multi-parameter correlation
        - High-accuracy requirements
        """)
    
    st.divider()
    
    st.markdown("### Monitored Parameters")
    
    params_data = {
        "Parameter": ["Altitude", "Airspeed", "Fuel Level", "Engine Temperature"],
        "Unit": ["feet", "knots", "pounds", "°Celsius"],
        "Normal Range": ["8,000 - 12,000", "230 - 270", "Variable", "180 - 220"],
        "Detection Sensitivity": ["±500 ft", "±50 kt", "Rate-based", "±30°C"]
    }
    
    st.dataframe(
        pd.DataFrame(params_data),
        hide_index=True,
        use_container_width=True
    )

# Footer
st.divider()
st.markdown(
    """
    <div style='text-align: center; color: #95a5a6; font-size: 0.9rem;'>
    <p>Flight Anomaly Detection System v1.0 | Developed with Python & Streamlit</p>
    <p>Statistical Analysis & Machine Learning Platform</p>
    </div>
    """,
    unsafe_allow_html=True
)
