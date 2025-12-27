import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import date, datetime, time
import logging
from utils import ensure_setup, log_new_sleep_entry, update_mood, predict_wakeup, CSV_PATH

# Configure logging for Streamlit
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Smart Alarm System",
    page_icon="⏰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 0.75rem;
        border-radius: 0.25rem;
        margin: 1rem 0;
    }
    .warning-message {
        background-color: #fff3cd;
        color: #856404;
        padding: 0.75rem;
        border-radius: 0.25rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False
if 'df' not in st.session_state:
    st.session_state.df = None
if 'models' not in st.session_state:
    st.session_state.models = None

# Main title
st.markdown('<h1 class="main-header">⏰ Smart Alarm System</h1>', unsafe_allow_html=True)

# Sidebar for system status and info
with st.sidebar :
    st.header("📊 System Status")

    # Initialize system
    if not st.session_state.models_loaded:
        with st.spinner("Loading models and data..."):
            try:
                st.session_state.df, st.session_state.models = ensure_setup()
                st.session_state.models_loaded = True
                st.success("✅ System ready!")
            except Exception as e:
                st.error(f"❌ System initialization failed: {e}")
                st.stop()

    # Display system stats
    if st.session_state.df is not None:
        st.metric("📈 Total Sleep Records", len(st.session_state.df))
        st.metric("📅 Data Range", f"{len(st.session_state.df)} days")

        # Recent mood distribution
        recent_moods = st.session_state.df.tail(30)['mood'].value_counts()
        if not recent_moods.empty:
            st.write("**Recent Mood (last 30 days):**")
            for mood, count in recent_moods.items():
                if mood:  # Skip empty moods
                    st.write(f"- {mood.title()}: {count}")

    # System controls
    st.header("⚙️ System Controls")
    if st.button("🔄 Refresh Data"):
        st.session_state.models_loaded = False
        st.rerun()

    if st.button("📊 Show Detailed Stats"):
        st.session_state.show_stats = True

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    # --- Bedtime logging section
    st.header("🌙 Set Smart Alarm")

    with st.container():
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)

        # Input validation and user guidance
        st.info("💡 **Tip:** The system automatically adds 15 minutes to account for falling asleep time.")

        # Time input with validation
        default_time = time(23, 0)  # Default to 11:00 PM
        sleep_time = st.time_input(
            "What time are you planning to go to bed?",
            value=default_time,
            help="Enter your planned bedtime. The system will automatically adjust for falling asleep time."
        )

        # Advanced options
        with st.expander("⚙️ Advanced Options"):
            buffer_minutes = st.slider(
                "Sleep buffer (minutes)",
                min_value=0,
                max_value=60,
                value=15,
                help="Extra time added to account for falling asleep"
            )

            show_prediction_details = st.checkbox(
                "Show prediction details",
                help="Display individual model predictions and confidence"
            )

        # Prediction preview
        if st.session_state.models_loaded:
            sleep_time_str = sleep_time.strftime("%H:%M")
            try:
                # Show live prediction as user changes time
                preview_result = predict_wakeup(st.session_state.models, sleep_time_str)

                col_pred1, col_pred2, col_pred3 = st.columns(3)
                with col_pred1:
                    st.metric(
                        "🎯 Predicted Wake Time",
                        preview_result['pred_meta_hhmm'],
                        help="Based on your historical sleep patterns"
                    )
                with col_pred2:
                    st.metric(
                        "💤 Expected Sleep Duration",
                        f"{preview_result['sleep_duration_hours']:.1f}h"
                    )
                with col_pred3:
                    confidence = "High" if abs(
                        preview_result['day_prediction'] - preview_result['week_prediction']) < 60 else "Medium"
                    st.metric("🎪 Prediction Confidence", confidence)

                # Show detailed predictions if requested
                if show_prediction_details:
                    st.write("**Model Details:**")
                    st.write(
                        f"- Day-based model: {int(preview_result['day_prediction'] // 60):02d}:{int(preview_result['day_prediction'] % 60):02d}")
                    st.write(
                        f"- Week-based model: {int(preview_result['week_prediction'] // 60):02d}:{int(preview_result['week_prediction'] % 60):02d}")
                    st.write(f"- Final prediction (average): {preview_result['pred_meta_hhmm']}")

            except Exception as e:
                st.warning(f"⚠️ Prediction preview unavailable: {e}")

        # Set alarm button
        if st.button("🚨 Set Smart Alarm", type="primary", use_container_width=True):
            if st.session_state.models_loaded:
                try:
                    with st.spinner("Setting your smart alarm..."):
                        new_row, prediction = log_new_sleep_entry(
                            CSV_PATH,
                            sleep_time.strftime("%H:%M"),
                            st.session_state.models,
                            buffer_minutes
                        )

                    # Success message with details
                    st.balloons()
                    st.markdown(f"""
                    <div class="success-message">
                        <h4>✅ Smart Alarm Set Successfully!</h4>
                        <ul>
                            <li><strong>Bedtime:</strong> {sleep_time.strftime('%H:%M')}</li>
                            <li><strong>Adjusted sleep time:</strong> {new_row['sleep_time']}</li>
                            <li><strong>Wake-up alarm:</strong> {prediction['pred_meta_hhmm']}</li>
                            <li><strong>Expected sleep:</strong> {prediction['sleep_duration_hours']:.1f} hours</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)

                    # Refresh data
                    st.session_state.models_loaded = False

                except Exception as e:
                    st.error(f"❌ Failed to set alarm: {e}")
            else:
                st.error("❌ System not ready. Please wait for initialization.")

        st.markdown('</div>', unsafe_allow_html=True)

    # --- Morning mood section
    st.header("☀️ Morning Check-in")

    with st.container():
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)

        today_str = date.today().isoformat()

        # Check if there's an entry for today
        if st.session_state.df is not None:
            today_entry = st.session_state.df[st.session_state.df['date'] == today_str]
            if not today_entry.empty:
                current_mood = today_entry.iloc[0]['mood'] if today_entry.iloc[0]['mood'] else None
                if current_mood:
                    st.info(
                        f"Current mood for today: **{current_mood.title()}** 😊" if current_mood == "fresh" else f"Current mood for today: **{current_mood.title()}** 😴")
            else:
                st.warning("No sleep entry found for today. Set an alarm first!")

        # Mood selection with emojis
        col_mood1, col_mood2 = st.columns(2)
        with col_mood1:
            if st.button("😊 Fresh & Energetic", use_container_width=True, type="secondary"):
                st.session_state.selected_mood = "fresh"

        with col_mood2:
            if st.button("😴 Tired & Sleepy", use_container_width=True, type="secondary"):
                st.session_state.selected_mood = "sleepy"

        # Save mood
        if 'selected_mood' in st.session_state:
            if st.button(f"💾 Save Mood: {st.session_state.selected_mood.title()}", type="primary",
                         use_container_width=True):
                try:
                    updated = update_mood(CSV_PATH, today_str, st.session_state.selected_mood)
                    st.success(f"✅ Mood updated to: **{st.session_state.selected_mood.title()}**")

                    # Show updated entry
                    with st.expander("📋 Updated Entry"):
                        st.json(updated)

                    # Clear selection and refresh
                    del st.session_state.selected_mood
                    st.session_state.models_loaded = False

                except Exception as e:
                    st.error(f"❌ Failed to update mood: {e}")

        st.markdown('</div>', unsafe_allow_html=True)

with col2:
    # --- Sleep analytics and history
    st.header("📊 Sleep Analytics")

    if st.session_state.df is not None and len(st.session_state.df) > 0:
        # Recent sleep history
        st.subheader("� Recent Sleep Log")
        recent_df = st.session_state.df.tail(7).copy()

        # Format the display
        recent_df['Sleep Duration'] = recent_df.apply(lambda row:
                                                      f"{((pd.to_datetime(row['wake_time'], format='%H:%M').hour * 60 + pd.to_datetime(row['wake_time'], format='%H:%M').minute) - (pd.to_datetime(row['sleep_time'], format='%H:%M').hour * 60 + pd.to_datetime(row['sleep_time'], format='%H:%M').minute)) / 60:.1f}h",
                                                      axis=1)

        display_df = recent_df[['date', 'sleep_time', 'wake_time', 'Sleep Duration', 'mood']].copy()
        display_df.columns = ['Date', 'Bedtime', 'Wake Time', 'Duration', 'Mood']

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )

        # Sleep pattern visualization
        if len(st.session_state.df) > 7:
            st.subheader("📈 Sleep Pattern Trends")

            # Convert times to minutes for plotting
            df_viz = st.session_state.df.tail(30).copy()
            df_viz['sleep_minutes'] = df_viz['sleep_time'].apply(lambda x:
                                                                 int(x.split(':')[0]) * 60 + int(x.split(':')[1]))
            df_viz['wake_minutes'] = df_viz['wake_time'].apply(lambda x:
                                                               int(x.split(':')[0]) * 60 + int(x.split(':')[1]))
            df_viz['duration'] = (df_viz['wake_minutes'] - df_viz['sleep_minutes']) % 1440 / 60

            # Create sleep duration chart
            fig = px.line(
                df_viz,
                x='date',
                y='duration',
                title='Sleep Duration Trend (Last 30 Days)',
                labels={'duration': 'Sleep Duration (hours)', 'date': 'Date'}
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)

        # Sleep quality metrics
        st.subheader("🎯 Sleep Quality Metrics")

        # Calculate metrics
        avg_sleep = st.session_state.df.tail(30).apply(lambda row:
                                                       ((pd.to_datetime(row['wake_time'], format='%H:%M').hour * 60 +
                                                         pd.to_datetime(row['wake_time'], format='%H:%M').minute) -
                                                        (pd.to_datetime(row['sleep_time'], format='%H:%M').hour * 60 +
                                                         pd.to_datetime(row['sleep_time'],
                                                                        format='%H:%M').minute)) / 60, axis=1).mean()

        fresh_ratio = len(st.session_state.df[st.session_state.df['mood'] == 'fresh']) / len(
            st.session_state.df[st.session_state.df['mood'] != '']) * 100 if len(
            st.session_state.df[st.session_state.df['mood'] != '']) > 0 else 0

        col_metric1, col_metric2 = st.columns(2)
        with col_metric1:
            st.metric("⏱️ Avg Sleep Duration", f"{avg_sleep:.1f}h")
        with col_metric2:
            st.metric("😊 Fresh Mornings", f"{fresh_ratio:.0f}%")

    else:
        st.info("📊 No sleep data available yet. Start by setting your first alarm!")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <small>🤖 Smart Alarm System v2.0 | Powered by Machine Learning | 
    <a href='#' onclick='st.rerun()'>Refresh Data</a></small>
</div>
""", unsafe_allow_html=True)
