import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
import xgboost as xgb
from prophet import prophet
import holidays
import plotly.express as px
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="ROAS Forecasting Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
st.title("🚀 ROAS Forecasting and Optimization Pipeline")
st.markdown("### Machine Learning-Powered Marketing Campaign Analytics")

# Sidebar
with st.sidebar:
    st.header("📊 Dashboard Controls")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Upload Marketing Campaign Dataset",
        type=['csv'],
        help="Upload your marketing campaign CSV file"
    )
    
    if uploaded_file:
        st.success("✅ File uploaded successfully!")

# Main content
if uploaded_file is None:
    # Landing page content
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="🎯 Prediction Accuracy Target",
            value="MAPE ≤ 20%",
            help="Mean Absolute Percentage Error target for ROAS predictions"
        )
    
    with col2:
        st.metric(
            label="🔮 Forecasting Horizon",
            value="30 Days",
            help="Time series forecasting using Facebook Prophet"
        )
    
    with col3:
        st.metric(
            label="🎨 Models Used",
            value="XGBoost + Prophet",
            help="Machine learning and time series models"
        )
    
    st.markdown("---")
    
    # Feature highlights
    st.subheader("🎯 Key Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **📈 ROAS Prediction Models**
        - XGBoost regression for Day 1, 7, and 30 ROAS
        - Advanced feature engineering (CAC, seasonality, etc.)
        - High accuracy predictions (MAPE ≤ 20%)
        
        **⏰ Time Series Forecasting**
        - Facebook Prophet for revenue forecasting
        - Holiday and seasonal effect modeling
        - Campaign-level trend analysis
        """)
    
    with col2:
        st.markdown("""
        **💰 Budget Optimization**
        - Portfolio-level ROAS uplift simulation
        - Channel reallocation recommendations
        - Data-driven budget scaling decisions
        
        **🎭 Interactive Dashboard**
        - Real-time model predictions
        - Campaign performance insights
        - Actionable recommendations
        """)
    
    st.markdown("---")
    st.info("👆 **Upload your marketing campaign dataset using the sidebar to get started!**")

else:
    # Process uploaded data
    try:
        # Load data
        df = pd.read_csv(uploaded_file)
        
        # Data preprocessing function
        @st.cache_data
        def prepare_data(df):
            df = df.copy()
            df['date'] = pd.to_datetime(df['date'])
            
            # Feature engineering
            df['CAC'] = df.apply(lambda x: x['spend'] / x['installs'] if x['installs'] > 0 else 0, axis=1)
            df['is_weekend'] = df['date'].dt.dayofweek >= 5
            
            # US holidays
            us_holidays = holidays.US()
            df['is_holiday'] = df['date'].apply(lambda x: x in us_holidays)
            
            # Encode categorical variables
            df['campaign_encoded'] = df['campaign'].astype('category').cat.codes
            df['channel_encoded'] = df['channel'].astype('category').cat.codes
            
            # Additional features
            df['spend_per_install'] = df['spend'] / (df['installs'] + 1)
            df['conversion_rate'] = df['conversions'] / (df['installs'] + 1)
            df['revenue_per_conversion'] = df['revenue'] / (df['conversions'] + 1)
            
            return df
        
        # Prepare data
        df_processed = prepare_data(df)
        
        # Sidebar options
        with st.sidebar:
            st.markdown("---")
            st.subheader("🔧 Analysis Options")
            
            analysis_type = st.selectbox(
                "Choose Analysis Type",
                ["📊 Data Overview", "🤖 ROAS Prediction", "📈 Time Series Forecast", "💰 Budget Simulator"]
            )
        
        # Main dashboard content based on selection
        if analysis_type == "📊 Data Overview":
            st.subheader("📊 Dataset Overview")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Total Records",
                    f"{len(df_processed):,}",
                    help="Number of campaign records in dataset"
                )
            
            with col2:
                st.metric(
                    "Campaigns",
                    f"{df_processed['campaign'].nunique():,}",
                    help="Unique campaigns in dataset"
                )
            
            with col3:
                st.metric(
                    "Channels",
                    f"{df_processed['channel'].nunique():,}",
                    help="Marketing channels analyzed"
                )
            
            with col4:
                st.metric(
                    "Date Range",
                    f"{(df_processed['date'].max() - df_processed['date'].min()).days} days",
                    help="Time span of the dataset"
                )
            
            # Data preview
            st.markdown("### 📋 Raw Data Preview")
            st.dataframe(df.head(10), use_container_width=True)
            
            # Summary statistics
            st.markdown("### 📈 Key Metrics Summary")
            summary_cols = ['spend', 'installs', 'conversions', 'revenue', 'roas_day_1', 'roas_day_7', 'roas_day_30']
            if all(col in df.columns for col in summary_cols):
                st.dataframe(df[summary_cols].describe(), use_container_width=True)
        
        elif analysis_type == "🤖 ROAS Prediction":
            st.subheader("🤖 ROAS Prediction Models")
            
            # Train models
            with st.spinner("🔄 Training XGBoost models..."):
                targets = ['roas_day_1', 'roas_day_7', 'roas_day_30']
                feature_cols = [
                    'spend', 'installs', 'conversions', 'CAC', 'is_weekend', 'is_holiday',
                    'campaign_encoded', 'channel_encoded', 'spend_per_install', 'conversion_rate', 'revenue_per_conversion'
                ]
                
                # Check if all required columns exist
                missing_cols = [col for col in targets + feature_cols if col not in df_processed.columns]
                
                if missing_cols:
                    st.error(f"❌ Missing required columns: {missing_cols}")
                    st.info("💡 Please ensure your dataset has the required columns for ROAS prediction.")
                else:
                    X = df_processed[feature_cols]
                    results = {}
                    
                    for target in targets:
                        y = df_processed[target]
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                        
                        model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
                        model.fit(X_train, y_train)
                        
                        y_pred = model.predict(X_test)
                        mape = mean_absolute_percentage_error(y_test, y_pred)
                        rmse = mean_squared_error(y_test, y_pred, squared=False)
                        
                        results[target] = {'MAPE': mape, 'RMSE': rmse, 'model': model}
                    
                    # Display results
                    st.markdown("### 🎯 Model Performance")
                    
                    perf_data = []
                    for target, metrics in results.items():
                        perf_data.append({
                            'Target': target.replace('_', ' ').title(),
                            'MAPE': f"{metrics['MAPE']:.4f}",
                            'RMSE': f"{metrics['RMSE']:.4f}",
                            'Status': "✅ Excellent" if metrics['MAPE'] <= 0.20 else "⚠️ Needs Improvement"
                        })
                    
                    st.dataframe(pd.DataFrame(perf_data), use_container_width=True, hide_index=True)
                    
                    # Feature importance
                    st.markdown("### 🔍 Feature Importance (ROAS Day 1)")
                    if 'roas_day_1' in results:
                        importance_df = pd.DataFrame({
                            'Feature': feature_cols,
                            'Importance': results['roas_day_1']['model'].feature_importances_
                        }).sort_values('Importance', ascending=False)
                        
                        fig = px.bar(
                            importance_df.head(10),
                            x='Importance',
                            y='Feature',
                            orientation='h',
                            title="Top 10 Most Important Features"
                        )
                        st.plotly_chart(fig, use_container_width=True)
        
        elif analysis_type == "📈 Time Series Forecast":
            st.subheader("📈 Time Series Forecasting with Prophet")
            
            # Select campaign and channel
            col1, col2 = st.columns(2)
            
            with col1:
                selected_campaign = st.selectbox(
                    "Select Campaign",
                    options=df_processed['campaign'].unique(),
                    help="Choose a campaign for forecasting"
                )
            
            with col2:
                selected_channel = st.selectbox(
                    "Select Channel",
                    options=df_processed['channel'].unique(),
                    help="Choose a marketing channel"
                )
            
            forecast_days = st.slider("Forecast Horizon (days)", 7, 60, 30)
            
            if st.button("🔮 Generate Forecast", type="primary"):
                with st.spinner("🔄 Generating Prophet forecast..."):
                    # Filter data
                    filtered_data = df_processed[
                        (df_processed['campaign'] == selected_campaign) &
                        (df_processed['channel'] == selected_channel)
                    ]
                    
                    if len(filtered_data) == 0:
                        st.warning("⚠️ No data found for selected campaign and channel combination.")
                    else:
                        # Prepare time series data
                        ts_data = filtered_data.groupby('date').agg({
                            'revenue': 'sum'
                        }).reset_index()
                        
                        ts_data = ts_data.rename(columns={'date': 'ds', 'revenue': 'y'})
                        
                        # Create and fit Prophet model
                        model = Prophet()
                        model.add_country_holidays(country_name='US')
                        model.fit(ts_data)
                        
                        # Make forecast
                        future = model.make_future_dataframe(periods=forecast_days)
                        forecast = model.predict(future)
                        
                        # Plot forecast
                        fig = go.Figure()
                        
                        # Historical data
                        fig.add_trace(go.Scatter(
                            x=ts_data['ds'],
                            y=ts_data['y'],
                            mode='markers+lines',
                            name='Historical Revenue',
                            line=dict(color='blue')
                        ))
                        
                        # Forecast
                        fig.add_trace(go.Scatter(
                            x=forecast['ds'],
                            y=forecast['yhat'],
                            mode='lines',
                            name='Forecasted Revenue',
                            line=dict(color='red', dash='dash')
                        ))
                        
                        # Confidence intervals
                        fig.add_trace(go.Scatter(
                            x=forecast['ds'],
                            y=forecast['yhat_upper'],
                            fill=None,
                            mode='lines',
                            line_color='rgba(0,100,80,0)',
                            showlegend=False
                        ))
                        
                        fig.add_trace(go.Scatter(
                            x=forecast['ds'],
                            y=forecast['yhat_lower'],
                            fill='tonexty',
                            mode='lines',
                            line_color='rgba(0,100,80,0)',
                            name='Confidence Interval'
                        ))
                        
                        fig.update_layout(
                            title=f'Revenue Forecast: {selected_campaign} - {selected_channel}',
                            xaxis_title='Date',
                            yaxis_title='Revenue ($)',
                            hovermode='x'
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Forecast summary
                        future_data = forecast[forecast['ds'] > ts_data['ds'].max()]
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric(
                                "Forecasted Revenue (30 days)",
                                f"${future_data['yhat'].sum():,.0f}"
                            )
                        
                        with col2:
                            st.metric(
                                "Average Daily Revenue",
                                f"${future_data['yhat'].mean():,.0f}"
                            )
                        
                        with col3:
                            growth_rate = ((future_data['yhat'].mean() / ts_data['y'].mean()) - 1) * 100
                            st.metric(
                                "Projected Growth",
                                f"{growth_rate:+.1f}%"
                            )
        
        elif analysis_type == "💰 Budget Simulator":
            st.subheader("💰 Budget Reallocation Simulator")
            
            st.markdown("### 🎛️ Adjust Channel Budgets")
            st.info("💡 Modify budget allocations by percentage. Positive values increase spend, negative values decrease spend.")
            
            channels = df_processed['channel'].unique()
            adjustments = {}
            
            col1, col2 = st.columns(2)
            
            for i, channel in enumerate(channels):
                with col1 if i % 2 == 0 else col2:
                    adjustments[channel] = st.slider(
                        f"{channel} Budget Change (%)",
                        min_value=-50,
                        max_value=50,
                        value=0,
                        step=5,
                        help=f"Adjust {channel} channel budget by percentage"
                    )
            
            if st.button("🧮 Run Simulation", type="primary"):
                with st.spinner("🔄 Running budget simulation..."):
                    # Simple simulation logic
                    df_sim = df_processed.copy()
                    
                    for channel, adjustment in adjustments.items():
                        mask = df_sim['channel'] == channel
                        df_sim.loc[mask, 'spend'] *= (1 + adjustment / 100)
                    
                    # Recalculate CAC
                    df_sim['CAC'] = df_sim.apply(
                        lambda x: x['spend'] / x['installs'] if x['installs'] > 0 else 0, 
                        axis=1
                    )
                    
                    # Calculate portfolio ROAS changes
                    original_portfolio_roas = (df_processed['roas_day_1'] * df_processed['spend']).sum() / df_processed['spend'].sum()
                    new_portfolio_roas = (df_sim['roas_day_1'] * df_sim['spend']).sum() / df_sim['spend'].sum()
                    
                    uplift = new_portfolio_roas - original_portfolio_roas
                    uplift_percentage = (uplift / original_portfolio_roas) * 100
                    
                    # Display results
                    st.markdown("### 📊 Simulation Results")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(
                            "Original Portfolio ROAS",
                            f"{original_portfolio_roas:.4f}",
                            help="Current portfolio-level ROAS"
                        )
                    
                    with col2:
                        st.metric(
                            "Projected Portfolio ROAS",
                            f"{new_portfolio_roas:.4f}",
                            delta=f"{uplift:+.4f}",
                            help="Projected ROAS after budget reallocation"
                        )
                    
                    with col3:
                        st.metric(
                            "ROAS Uplift",
                            f"{uplift_percentage:+.2f}%",
                            help="Percentage improvement in portfolio ROAS"
                        )
                    
                    # Recommendations
                    if uplift > 0:
                        st.success(f"✅ **Positive Impact**: The proposed budget reallocation could increase portfolio ROAS by {uplift_percentage:+.2f}%")
                    elif uplift < 0:
                        st.warning(f"⚠️ **Negative Impact**: The proposed budget reallocation could decrease portfolio ROAS by {uplift_percentage:.2f}%")
                    else:
                        st.info("ℹ️ **Neutral Impact**: The proposed budget reallocation has minimal impact on portfolio ROAS")
    
    except Exception as e:
        st.error(f"❌ Error processing uploaded file: {str(e)}")
        st.info("💡 Please ensure your CSV file has the required columns: date, campaign, channel, spend, installs, conversions, revenue, roas_day_1, roas_day_7, roas_day_30")

# Footer
st.markdown("---")
st.markdown("### 🎯 About This Dashboard")
st.markdown("""
This **ROAS Forecasting and Optimization Pipeline** demonstrates advanced analytics capabilities including:
- **Machine Learning**: XGBoost regression models for accurate ROAS prediction
- **Time Series Forecasting**: Facebook Prophet for revenue forecasting with seasonality
- **Business Intelligence**: Budget optimization simulator for portfolio-level decisions
- **Interactive Analytics**: Real-time dashboard for marketing campaign insights

**Built with**: Python, Streamlit, XGBoost, Prophet, Plotly
""")

st.markdown("*🚀 Deployed on Streamlit Community Cloud*")

