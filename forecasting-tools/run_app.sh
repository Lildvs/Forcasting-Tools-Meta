#!/bin/bash

# Make sure the data directory exists
mkdir -p data

# Run the Streamlit app
streamlit run streamlit_app.py "$@" 