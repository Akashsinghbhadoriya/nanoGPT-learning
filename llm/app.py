import streamlit as st

# 1. Page Title & Headers
st.title("Large Language Model Visualizer")
st.header("This contains the visualizer to view various implementations of the transformer architecture and run them")

# 2. Add Interactive Widgets
name = st.text_input("What is your name?")
age = st.slider("Select your age", 0, 100, 25)

# 3. Handle User Input
if st.button("Submit"):
    st.success(f"Hello {name}! You are {age} years old.")
