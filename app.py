import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

#  Page Configuration 
st.set_page_config(
    page_title="T20 World Cup Analytics",
    layout="wide",
    initial_sidebar_state="collapsed"
)

#  THEME
st.markdown("""
<style>
    /* IMPORT FONT */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');

    /* GENERAL SETTINGS */
    * {
        font-family: 'Poppins', sans-serif;
    }
    
    /* BACKGROUND */
    .stApp {
        background: linear-gradient(to bottom right, #0f172a, #1e293b);
        color: #e2e8f0;
    }

    /* CUSTOM TITLE HEADER */
    .title-card {
        background: linear-gradient(90deg, #3b82f6 0%, #2563eb 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }

    /* CUSTOM TABS */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background-color: transparent;
        padding-bottom: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 6px;
        color: #94a3b8;
        padding: 10px 25px;
        transition: all 0.3s ease;
    }
    .stTabs [aria-selected="true"] {
        background-color: #3b82f6 !important; /* Electric Blue */
        color: white !important;
        border-color: #3b82f6 !important;
        box-shadow: 0 0 15px rgba(59, 130, 246, 0.5);
    }

    /* RESULT CARDS (Glassmorphism) */
    .result-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        margin-top: 20px;
        backdrop-filter: blur(10px);
    }
    
    /* METRIC HIGHLIGHTS */
    .metric-value {
        font-size: 2.5rem;
        font-weight: 600;
        color: #60a5fa;
    }
    
    /* INPUT FIELDS */
    .stSelectbox div[data-baseweb="select"] {
        background-color: #1e293b !important;
        color: white !important;
        border-color: #475569 !important;
    }
    
    /* BUTTONS */
    .stButton button {
        background: linear-gradient(90deg, #3b82f6 0%, #2563eb 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 8px;
        font-weight: 600;
        transition: transform 0.2s;
        width: 100%;
    }
    .stButton button:hover {
        transform: scale(1.02);
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.5);
    }

    /* FOOTER */
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #0f172a;
        color: #94a3b8;
        text-align: center;
        padding: 10px;
        font-size: 12px;
        border-top: 1px solid #1e293b;
    }
</style>
<div class="footer">
    <p>Developed by <b>Sakshi Shukla</b> | Advanced T20 Analytics Engine</p>
</div>
""", unsafe_allow_html=True)

#  Load Resources 
@st.cache_resource
def load_resources():
    try:
        with open('cricket_models.pkl', 'rb') as f:
            artifacts = pickle.load(f)
        return artifacts
    except FileNotFoundError:
        return None

@st.cache_data
def load_dataframe():
    try:
        return pd.read_csv('cricket_ml_ready.csv')
    except FileNotFoundError:
        return pd.DataFrame()

# Load everything
artifacts = load_resources()
df = load_dataframe()

#  4. App Layout 
st.markdown('<div class="title-card"><h1> T20 World Cup Predictor</h1><p></p></div>', unsafe_allow_html=True)

if artifacts is None:
    st.error(" Model file 'cricket_models.pkl' not found! Please run your training notebook to generate it.")
    st.stop()

# Unpack the specific models and helpers from the pickle file
score_pipeline = artifacts['score_model_pipeline']
win_model = artifacts['win_model']
win_encoders = artifacts['win_encoders']
win_context = artifacts['win_context']

tab_pred, tab_analytics, tab_about = st.tabs([" Match Predictor", " Deep Analytics", " Model Info"])

#  TAB 1: SIMULATOR 
with tab_pred:
    st.markdown("### Match Configuration")
    col1, col2, col3 = st.columns(3)
    
    # Use the classes from the encoders to ensure dropdowns match training data exactly
    teams = sorted(win_encoders['batting_team'].classes_)
    venues = sorted(win_encoders['venue'].classes_)

    with col1:
        batting_team = st.selectbox("Batting Team", teams)
    with col2:
        bowling_team = st.selectbox("Bowling Team", teams, index=1)
    with col3:
        venue = st.selectbox("Venue", venues)

    st.markdown("---")

    # Scenario Toggle
    match_status = st.radio("Select Game State:", 
                            [" Batting First (Setting Target)", " Chasing (Second Innings)"], 
                            horizontal=True)

    col_in1, col_in2, col_in3, col_in4 = st.columns(4)
    with col_in1:
        current_score = st.number_input("Current Runs", min_value=0, step=1)
    with col_in2:
        overs_done = st.number_input("Overs Done", min_value=0.0, max_value=20.0, step=0.1)
    with col_in3:
        wickets = st.number_input("Wickets Fallen", min_value=0, max_value=10, step=1)
    with col_in4:
        is_rain = st.checkbox("Rain Interruption?")

    # Dynamic Calculations
    balls_bowled = int(overs_done * 6)
    balls_left = 120 - balls_bowled
    wickets_left = 10 - wickets
    # Avoid division by zero for CRR
    crr = current_score / overs_done if overs_done > 0 else 0
    
    target = 0
    runs_needed = 0
    rrr = 0
    
    if match_status == " Chasing (Second Innings)":
        target = st.number_input("Target to Chase", min_value=current_score, value=current_score+1)
        runs_needed = target - current_score
        if balls_left > 0:
            rrr = (runs_needed / balls_left) * 6
        else:
            rrr = 99 # High RRR if no balls left
    
    st.write("") # Spacer
    
    if st.button("RUN PREDICTION"):
        try:
            #  SCENARIO 1: SCORE PREDICTION (Gradient Boosting Pipeline) 
            if match_status == " Batting First (Setting Target)":
                # The Pipeline expects a DataFrame with raw column names
                input_data = pd.DataFrame({
                    'batting_team': [batting_team],
                    'bowling_team': [bowling_team],
                    'venue': [venue],
                    'current_score': [current_score],
                    'balls_left': [balls_left],
                    'wickets_left': [wickets_left],
                    'crr': [crr],
                    'is_rain': [1 if is_rain else 0]
                })
                
                prediction = score_pipeline.predict(input_data)[0]
                
                # Result Card
                st.markdown(f"""
                <div class="result-card">
                    <h3>Projected First Innings Score</h3>
                    <div class="metric-value">{int(prediction)}</div>
                    <p>Current Rate: <b>{crr:.2f}</b> | Projected: <b>{int(prediction)}</b></p>
                </div>
                """, unsafe_allow_html=True)

            #  SCENARIO 2: WIN PREDICTION (Random Forest + Context) 
            else:
                # 1. Encode Inputs Manually (because RF doesn't have a pipeline)
                def safe_encode(encoder, value):
                    if value in encoder.classes_:
                        return encoder.transform([value])[0]
                    return -1 # Fallback for unknown
                
                enc_bat = safe_encode(win_encoders['batting_team'], batting_team)
                enc_bowl = safe_encode(win_encoders['bowling_team'], bowling_team)
                enc_venue = safe_encode(win_encoders['venue'], venue)
                
                # 2. Get Context (Venue Bias)
                # Look up the bias from our loaded dictionary
                venue_bias_val = win_context['venue_bias'].get(venue, win_context['global_bias'])
                
                # 3. Create Input Array
                # Order: ['batting_team', 'bowling_team', 'venue', 'runs_needed', 'balls_left', 'wickets_left', 'crr', 'rrr', 'venue_bias', 'is_rain']
                input_data = [[enc_bat, enc_bowl, enc_venue, runs_needed, balls_left, wickets_left, crr, rrr, venue_bias_val, 1 if is_rain else 0]]
                
                prob = win_model.predict_proba(input_data)[0]
                win_prob_batting = prob[1] 
                win_prob_bowling = prob[0] 

                col_res1, col_res2 = st.columns(2)
                with col_res1:
                    st.markdown(f"""
                    <div class="result-card" style="border-left: 4px solid #4ade80;">
                        <h4>{batting_team}</h4>
                        <div class="metric-value" style="color: #4ade80;">{win_prob_batting*100:.1f}%</div>
                        <p>Win Probability</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_res2:
                    st.markdown(f"""
                    <div class="result-card" style="border-left: 4px solid #f87171;">
                        <h4>{bowling_team}</h4>
                        <div class="metric-value" style="color: #f87171;">{win_prob_bowling*100:.1f}%</div>
                        <p>Win Probability</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                if balls_left > 0:
                    st.info(f"Equation: {runs_needed} runs off {balls_left} balls | RRR: {rrr:.2f}")

        except Exception as e:
            st.error(f"Prediction Error: {e}")

#  TAB 2: ANALYTICS 
with tab_analytics:
    # Set Matplotlib dark theme
    plt.style.use('dark_background')
    
    st.markdown("### Interactive Data Analytics")
    
    if not df.empty:
        analysis_type = st.selectbox(
            "Select Analysis Component:",
            [
                "1. Venue Bias (Bat vs Chase)",
                "2. Scoring Trends (Histogram)",
                "3. High Scoring Venues",
                "4. Feature Correlations"
            ]
        )
        st.markdown("---")

        #  GRAPH 1: Venue Stats 
        if analysis_type == "1. Venue Bias (Bat vs Chase)":
            st.markdown("#### Venue Outcome Analysis")
            
            st.markdown("""
            <div style="background-color: #1e293b; padding: 10px; border-radius: 5px; border: 1px solid #334155;">
                <span style='color:#60a5fa; font-weight:bold;'>● Blue:</span> Chasing Wins &nbsp;&nbsp; 
                <span style='color:#f87171; font-weight:bold;'>● Red:</span> Batting First Wins
            </div><br>
            """, unsafe_allow_html=True)
            
            unique_venues = sorted(df['venue'].unique())
            selected_venue = st.selectbox("Select Stadium:", unique_venues)
            
            venue_data = df[(df['venue'] == selected_venue) & (df['inning'] == 2)].drop_duplicates(subset='match_id', keep='last')
            
            if not venue_data.empty:
                chase_wins = venue_data[venue_data['result_label'] == 1].shape[0]
                defend_wins = venue_data[venue_data['result_label'] == 0].shape[0]
                
                fig, ax = plt.subplots(figsize=(6, 6))
                fig.patch.set_alpha(0)
                ax.patch.set_alpha(0)
                
                labels = ['Chasing Won', 'Batting 1st Won']
                sizes = [chase_wins, defend_wins]
                colors = ['#60a5fa', '#f87171'] # Custom Blue/Red
                
                if sum(sizes) > 0:
                    wedges, texts, autotexts = ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
                    for text in texts: text.set_color('white')
                    for autotext in autotexts: autotext.set_color('white')
                    
                    st.pyplot(fig)
                else:
                    st.info("No match results available for this venue.")
            else:
                st.info("No data available.")

        #  GRAPH 2: Distribution 
        elif analysis_type == "2. Scoring Trends (Histogram)":
            st.markdown("#### Global Scoring Distribution")
            
            venues_list = ["All Venues"] + sorted(df['venue'].unique())
            selected_venue_dist = st.selectbox("Filter by Venue:", venues_list)
            
            dist_data = df[df['inning'] == 1].drop_duplicates(subset='match_id')
            if selected_venue_dist != "All Venues":
                dist_data = dist_data[dist_data['venue'] == selected_venue_dist]
                
            if not dist_data.empty:
                fig, ax = plt.subplots(figsize=(10, 5))
                fig.patch.set_alpha(0)
                ax.patch.set_alpha(0)
                
                sns.histplot(dist_data['final_score'], bins=20, kde=True, color='#2dd4bf', ax=ax)
                
                ax.set_xlabel("1st Innings Score", color='white')
                ax.set_ylabel("Frequency", color='white')
                ax.tick_params(colors='white')
                ax.grid(color='#334155', linestyle='--', linewidth=0.5)
                
                st.pyplot(fig)

        #  GRAPH 3: Average Score 
        elif analysis_type == "3. High Scoring Venues":
            st.markdown("#### Top Scoring Venues")
            
            top_n = st.slider("Show Top N Venues:", 5, 20, 10)
            avg_scores = df[df['inning'] == 1].groupby('venue')['final_score'].mean().sort_values(ascending=False).head(top_n)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            fig.patch.set_alpha(0)
            ax.patch.set_alpha(0)
            
            sns.barplot(x=avg_scores.values, y=avg_scores.index, palette="viridis", ax=ax)
            
            ax.set_xlabel("Average Score", color='white')
            ax.tick_params(colors='white')
            ax.grid(axis='x', color='#334155', linestyle='--', linewidth=0.5)
            
            st.pyplot(fig)

        # --- GRAPH 4: Heatmap ---
        elif analysis_type == "4. Feature Correlations":
            st.markdown("#### Statistical Correlation Matrix")
            
            numeric_df = df.select_dtypes(include=[np.number])
            if 'match_id' in numeric_df.columns:
                numeric_df = numeric_df.drop(columns=['match_id'])
            
            if not numeric_df.empty:
                fig, ax = plt.subplots(figsize=(10, 8))
                fig.patch.set_alpha(0)
                ax.patch.set_alpha(0)
                
                sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
                
                cbar = ax.collections[0].colorbar
                cbar.ax.tick_params(colors='white')
                ax.tick_params(colors='white')
                
                st.pyplot(fig)

#  TAB 3: ABOUT 
with tab_about:
    st.markdown("""
    <div class="result-card">
        <h2> About the Project</h2>
        <p>A two-model approach that separately predicts scores and match winners for clearer insights.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col_about1, col_about2 = st.columns(2)
    
    with col_about1:
        st.markdown("""
        ###  Model 1: Score Predictor
        **Algorithm:** Gradient Boosting Regressor
        
        * **Purpose:** Predicts 1st Innings Totals
        * **RMSE:** ~25.37 Runs
        * **Key Insight:** Uses One-Hot Encoding within a Pipeline to capture the exact impact of team matchups without data leakage.
        """)
        
    with col_about2:
        st.markdown("""
        ###  Model 2: Win Predictor
        **Algorithm:** Random Forest Classifier
        
        * **Purpose:** Predicts Match Outcome (Win/Loss)
        * **Accuracy:** ~74%
        * **Key Insight:** Enhanced with 'Venue Bias' and 'Required Run Rate' features to understand pressure situations.
        """)

    st.markdown("---")
    st.markdown("""
    ###  Data Pipeline
    1.  **Source:** Raw Ball-by-Ball JSON data from `cricsheet.org`.
    2.  **Processing:** * Converted JSON to structured CSV.
        * **Geocoding:** Mapped stadium names to exact Latitude/Longitude.
        * **Weather Integration:** Used **Meteostat API** to fetch historical rain and temperature data for specific match hours.
    
    ###  Developer
    **Sakshi Shukla** *B.Tech Mechanical Engineering | Data Science Enthusiast*
    """)