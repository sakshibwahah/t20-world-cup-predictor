# T20 World Cup Analytics and Prediction Engine

## Project Overview
This project is an advanced machine learning application designed to analyze and predict the outcomes of T20 International cricket matches. Developed as a independent individual engineering project, the system utilizes a hybrid machine learning architecture to provide real-time decision support and predictive analytics.

The application serves two primary functions: predicting first-innings scores based on current match progression and forecasting match winners by analyzing historical venue data and live game contexts.

## Live Deployment
**Access the live dashboard here:** [https://t20-world-cup-predictor-mywd9h8s43atddrsfjh7p3.streamlit.app/](https://t20-world-cup-predictor-mywd9h8s43atddrsfjh7p3.streamlit.app/)

## Technical Architecture
The system is built upon a dual-model framework to ensure high accuracy across different prediction tasks:

1.  **Score Prediction (Regression):**
    * **Algorithm:** Gradient Boosting Regressor.
    * **Implementation:** Encapsulated within a Scikit-Learn Pipeline to automate preprocessing (One-Hot Encoding for categorical variables and Standard Scaling for numerical inputs).
    * **Objective:** Predicts the projected final score based on current runs, wickets, and overs.

2.  **Win Prediction (Classification):**
    * **Algorithm:** Random Forest Classifier.
    * **Implementation:** Enhanced with a custom "Venue Bias" feature injection layer.
    * **Objective:** Calculates win probability for the chasing team by analyzing historical venue data, Required Run Rate (RRR), and Current Run Rate (CRR).

## Technology Stack

* **Programming Language:** Python 3.11.4
* **Frontend Framework:** Streamlit
* **Machine Learning:** Scikit-Learn (Random Forest, Gradient Boosting)
* **Data Manipulation:** Pandas, NumPy
* **Visualization:** Matplotlib, Seaborn
* **Deployment:** Streamlit Cloud

## Key Features

### 1. Match Outcome Predictor
A real-time simulation tool that allows users to input current match conditions (team, venue, score, wickets, overs). The model outputs the probability of victory for both the batting and bowling teams, adjusting for specific stadium characteristics (e.g., chasing bias at Wankhede Stadium).

### 2. Score Simulator
A predictive engine that estimates the final first-innings total. It analyzes the current run rate and wickets lost to project a realistic target, helping users understand if a team is performing above or below par.

### 3. Deep Analytics Dashboard
An interactive suite of visualizations providing insights into:
* Global scoring trends and distributions.
* Venue-specific outcome analysis (Batting First vs. Chasing).
* Correlation matrices for feature importance analysis.

## Repository Structure

* **app.py:** The main entry point for the Streamlit application. Contains the frontend logic and model inference calls.
* **cricket_models.pkl:** The serialized machine learning models and associated encoders.
* **requirements.txt:** A comprehensive list of Python dependencies required to run the project.
* **master_cricket_data.csv:** The processed dataset used for training the models.
* **model_training.ipynb:** The Jupyter Notebook containing the data cleaning, feature engineering, and model training source code.

## Setup and Installation

To run this project locally on your machine, follow these steps:

### Prerequisites
Ensure you have Python installed. It is recommended to use a virtual environment.

### Installation Steps

1.  **Clone the Repository**
    ```bash
    git clone [https://github.com/sakshibwahah/t20-world-cup-predictor.git](https://github.com/sakshibwahah/t20-world-cup-predictor.git)
    cd t20-world-cup-predictor
    ```

2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Application**
    ```bash
    streamlit run app.py
    ```

4.  **Access the Dashboard**
    The application will launch automatically in your default web browser at `http://localhost:8501`.

## User Guide

### Using the Match Predictor
1.  Navigate to the **Match Predictor** tab.
2.  Select the Batting Team, Bowling Team, and Venue.
3.  Enter the current match state (Runs, Wickets Fallen).
4.  Input the Overs and Balls completed separately (e.g., input '5' for Overs and '2' for Balls).
5.  If simulating a second innings scenario, select "Chasing" and input the Target.
6.  Click "Run Prediction" to view the Projected Score (1st Innings) or Win Probability (2nd Innings).

## Developer
**Sakshi Shukla**
B.Tech Mechanical Engineering
