# AI-Based Network Intrusion Detection System

## Project Description
This project implements an AI-Based Network Intrusion Detection System (NIDS) using Machine Learning.
It classifies network traffic as Normal or Intrusion using a Random Forest algorithm.
The system provides an interactive dashboard built with Streamlit.

## Features
- Machine Learning–based intrusion detection
- Random Forest classifier
- Live traffic simulation
- Interactive Streamlit dashboard
- Real-time prediction (Normal / Intrusion)

## Technologies Used
- Python
- Scikit-learn
- Pandas, NumPy
- Streamlit
- Matplotlib

## Dataset
- Simulated network traffic data (for demonstration)
- Can be extended to CIC-IDS2017 dataset

## How to Run the Project
```bash
py -3.11 -m pip install streamlit pandas numpy scikit-learn matplotlib
py -3.11 -m streamlit run nids_main.py
