import streamlit as st
import requests
import sys
import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import time
from datetime import datetime

# Ensure Python can find our custom modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.agents.copilot import OptimaHRCopilot

# ----------------------------------------------------
# 🌟 Ultra-Premium Glassmorphism CSS Injection
# ----------------------------------------------------
st.set_page_config(page_title="OptimaHR Ecosystem", page_icon="⚡", layout="wide")

def apply_custom_css():
    st.markdown("""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
            
            html, body, [class*="css"] {
                font-family: 'Outfit', sans-serif !important;
            }
            .stApp {
                background: radial-gradient(circle at top left, #121A28, #0C0F15);
                color: #FFFFFF;
            }
            
            /* Gradient Animated Text */
            .main-title {
                font-size: 55px;
                font-weight: 800;
                background: linear-gradient(-45deg, #FF6B6B, #4FACFE, #00F2FE, #FF0844);
                background-size: 300%;
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                animation: animated_text 5s ease-in-out infinite;
                margin-bottom: -10px;
            }
            
            @keyframes animated_text {
                0% { background-position: 0% 50%; }
                50% { background-position: 100% 50%; }
                100% { background-position: 0% 50%; }
            }

            .sub-title {
                font-size: 20px;
                color: #8B9BB4;
                font-weight: 300;
                margin-bottom: 40px;
            }

            /* Custom Glassmorphism Containers */
            div.css-1r6slb0, div.css-12oz5g7 {
                background: rgba(30, 41, 59, 0.4) !important;
                backdrop-filter: blur(10px);
                -webkit-backdrop-filter: blur(10px);
                border: 1px solid rgba(255, 255, 255, 0.05);
                border-radius: 16px;
                box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
            }

            /* Buttons */
            div.stButton > button {
                background: linear-gradient(90deg, #4FACFE, #00F2FE);
                color: white;
                font-weight: 600;
                transition: transform 0.2s ease, box-shadow 0.2s ease;
                border: none;
                border-radius: 8px;
            }
            div.stButton > button:hover {
                transform: translateY(-2px);
                box-shadow: 0px 8px 15px rgba(79, 172, 254, 0.4);
            }

            /* Chat Bubbles */
            .chat-box {
                background: rgba(45, 55, 72, 0.5);
                border-left: 5px solid #4FACFE;
                padding: 20px;
                border-radius: 0px 12px 12px 0px;
                font-size: 16px;
                font-weight: 300;
                color: #E2E8F0;
                box-shadow: inset 0px 0px 20px rgba(0,0,0,0.5);
                margin-top: 15px;
            }
            
            .agent-name {
                font-weight: 800;
                color: #4FACFE;
                font-size: 14px;
                text-transform: uppercase;
                letter-spacing: 1px;
                margin-bottom: 5px;
            }
        </style>
    """, unsafe_allow_html=True)

apply_custom_css()

# ----------------------------------------------------
# 📌 Dashboard Layout & Tabs
# ----------------------------------------------------
st.markdown('<div class="main-title">OptimaHR Ecosystem ⚡</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">AI-Powered Predictive & Prescriptive Analytics Engine</div>', unsafe_allow_html=True)

tab1, tab2 = st.tabs(["🔮 Copilot Prescriptive Engine", "🔴 Real-Time Global Telemetry"])

# ==========================================
# TAB 1: COPILOT ENGINE (Radar Chart & Insights)
# ==========================================
with tab1:
    col1, col2 = st.columns([1, 1.3])

    with col1:
        st.markdown("### 📥 Input Employee Metadata")
        
        with st.form("employee_form"):
            dept_c, level_c = st.columns(2)
            department = dept_c.selectbox("Department", ["Engineering", "Sales", "Marketing", "Customer Support", "HR"])
            job_level = level_c.selectbox("Job Level", ["Junior", "Mid", "Senior", "Lead"])
            
            exp_years = st.slider("Experience (Years)", 0.5, 20.0, 3.0)
            
            st.markdown("---")
            st.markdown("### 📊 Trajectory Metrics (Last 6 Months)")
            
            c1, c2 = st.columns(2)
            ot_rate = c1.number_input("On-Time Delivery Rate", min_value=0.0, max_value=1.0, value=0.6)
            login_hrs = c2.number_input("Avg Login Hours/Day", min_value=1.0, max_value=14.0, value=7.5)
            
            c3, c4 = st.columns(2)
            bugs = c3.number_input("Defect/Bug Count", min_value=0, max_value=50, value=12)
            training = c4.number_input("Training Hours", min_value=0, max_value=100, value=5)
            
            c5, c6 = st.columns(2)
            peer_score = c5.number_input("Peer Score (1-5)", min_value=1.0, max_value=5.0, value=3.0)
            mgr_score = c6.number_input("Manager Score (1-5)", min_value=1.0, max_value=5.0, value=3.0)
                
            submit_button = st.form_submit_button(label="⚡ Run Cortex Deep Analysis")

    with col2:
        if submit_button:
            # Prepare JSON payload for FastAPI Backend
            payload = {
                "department": department,
                "job_level": job_level,
                "experience_years": exp_years,
                "on_time_delivery_rate": ot_rate,
                "bug_count": bugs,
                "training_hours": training,
                "avg_login_hours": login_hrs,
                "peer_score": peer_score,
                "manager_score": mgr_score
            }
            
            with st.spinner("Initializing Multi-Agent Inference Server..."):
                try:
                    response = requests.post("http://localhost:8000/predict", json=payload)
                    if response.status_code == 200:
                        data = response.json()
                        predicted_band = data['predicted_band']
                        confidences = data['confidence_scores']
                        
                        st.markdown("### 🧠 AI Prediction Output")
                        
                        # Dynamic UI Metric
                        color = "#00F2FE" if predicted_band == "High" else "#F5A623" if predicted_band == "Medium" else "#FF6B6B"
                        st.markdown(f"<h2>Forecasted Rating: <span style='color:{color}'>{predicted_band}</span></h2>", unsafe_allow_html=True)
                        
                        # --- RADAR CHART: Employee vs Baseline ---
                        st.markdown("#### 🎯 Skills Signature vs Department Baseline")
                        categories = ['Delivery Rate', 'Login Engagement', 'Manager Score', 'Peer Score', 'Efficiency (Inverse Bugs)']
                        
                        # Normalize values to a 0-10 scale for the radar chart
                        emp_values = [
                            ot_rate * 10, 
                            (login_hrs / 12) * 10, 
                            (mgr_score / 5) * 10, 
                            (peer_score / 5) * 10, 
                            max(0, 10 - (bugs/2))
                        ]
                        # Mock baseline data for "Mid" Engineers
                        baseline = [8.5, 7.0, 8.0, 7.5, 6.0] 
                        
                        fig = go.Figure()
                        fig.add_trace(go.Scatterpolar(r=baseline, theta=categories, fill='toself', name='Dept Baseline', line_color='rgba(255,255,255,0.3)'))
                        fig.add_trace(go.Scatterpolar(r=emp_values, theta=categories, fill='toself', name='This Employee', line_color=color))
                        fig.update_layout(
                            polar=dict(radialaxis=dict(visible=True, range=[0, 10], color='rgba(255,255,255,0.2)'), bgcolor='rgba(0,0,0,0)'),
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                            font=dict(color='white'),
                            margin=dict(l=20, r=20, t=20, b=20)
                        )
                        st.plotly_chart(fig, use_container_width=True)

                        # --- COPILOT AGENT ---
                        st.markdown("### 🤖 Autonomous Copilot")
                        copilot = OptimaHRCopilot()
                        advice = copilot.get_prescriptive_advice(payload, predicted_band)
                        
                        html_chat = f"""
                        <div class="chat-box">
                            <div class="agent-name">HR Strategy Agent</div>
                            {advice.replace(chr(10), '<br>')}
                        </div>
                        """
                        st.markdown(html_chat, unsafe_allow_html=True)
                        
                    else:
                        st.error(f"Error from API Server.")
                except Exception as e:
                    st.error("🚨 Fast API Server is down! Run 'python main.py' to start the ecosystem.")
        else:
            st.info("👈 Enter employee telemetry and execute analysis to power up the AI.")


# ==========================================
# TAB 2: REAL-TIME GLOBAL TELEMETRY
# ==========================================
with tab2:
    st.markdown("### 🌐 Live Corporate Pulse feed")
    st.markdown("Simulating real-time WebSocket ingestion of employee activity metrics...")

    placeholder = st.empty()
    
    # We create a button to start simulation because infinite loops block Streamlit
    if st.button("▶️ Start Live Telemetry Feed"):
        
        # Initialize mock streaming data
        times = []
        commit_volume = []
        slack_sentiment = []
        
        for _ in range(50):
            with placeholder.container():
                c1, c2, c3 = st.columns(3)
                
                # Mock realtime metrics
                curr_time = datetime.now().strftime("%H:%M:%S")
                mock_anomalies = np.random.randint(0, 5)
                mock_login_rate = np.random.uniform(85, 99)
                mock_sentiment = round(np.random.normal(0.6, 0.1), 2)
                mock_commits = np.random.randint(100, 300)
                
                # Update Charts array
                times.append(curr_time)
                commit_volume.append(mock_commits)
                slack_sentiment.append(mock_sentiment)
                
                # Keep arrays bounded
                if len(times) > 15:
                    times.pop(0)
                    commit_volume.pop(0)
                    slack_sentiment.pop(0)
                
                # Cards
                c1.metric("Current Active Workforce Login %", f"{mock_login_rate:.1f}%", f"{np.random.choice([-1, 1]) * 0.5}%")
                c2.metric("Detected Risk Anomalies (AI)", mock_anomalies, f"+{mock_anomalies}")
                c3.metric("Company Wide Slack Sentiment", f"{mock_sentiment}", "Stable")
                
                st.markdown("---")
                colA, colB = st.columns(2)
                
                with colA:
                    st.markdown("#### 💻 Real-Time Global Commit Volume")
                    df_commits = pd.DataFrame({'Time': times, 'Commits': commit_volume})
                    fig_c = px.line(df_commits, x='Time', y='Commits', markers=True, color_discrete_sequence=['#00F2FE'])
                    fig_c.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
                    fig_c.update_xaxes(showgrid=False)
                    fig_c.update_yaxes(gridcolor='rgba(255,255,255,0.1)')
                    st.plotly_chart(fig_c, use_container_width=True)

                with colB:
                    st.markdown("#### 💬 Real-Time Employee Sentiment Index")
                    df_sent = pd.DataFrame({'Time': times, 'Sentiment': slack_sentiment})
                    fig_s = px.area(df_sent, x='Time', y='Sentiment', color_discrete_sequence=['#FF6B6B'])
                    fig_s.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
                    fig_s.update_xaxes(showgrid=False)
                    fig_s.update_yaxes(gridcolor='rgba(255,255,255,0.1)', range=[0, 1])
                    st.plotly_chart(fig_s, use_container_width=True)

            time.sleep(1) # Stream interval
            
    else:
        st.info("Click 'Start' to begin real-time enterprise emulation tracking.")
