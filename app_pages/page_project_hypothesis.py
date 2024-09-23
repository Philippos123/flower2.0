import streamlit as st
import matplotlib.pyplot as plt
import os

def page_project_hypothesis_body():
    st.write("### Project Hypotheses and Validation")

    st.success(
        f"* We hypothesized that infected leaves and healthy leaves would show clear visual signs of difference. \n\n"
        f"* Below is an image montage that demonstrates the differences we found after running the machine learning model on the dataset.\n"
        f"* Underneath the montage, we will explain the results and highlight the visual differences between healthy and infected leaves."
    )

    st.write("#### Analysis of an Average Healthy Leaf")

    # Ensure the image is in the correct location, and relative path is correct.
    st.image('outputs/v1/avg_var_healthy.png', caption="Average Healthy Leaf")

    st.write("#### Analysis of an Average Powdery Mildew Infected Leaf")

    # Ensure the image is in the correct location, and relative path is correct.
    st.image('outputs/v1/avg_var_powdery_mildew.png', caption="Average Infected Leaf")

    st.success(
        f"#### Observations\n"
        f"* There is a noticeable difference between healthy and infected leaves.\n"
        f"* On the black-background images, the white lines clearly outline the structure of a healthy leaf, while the lines are more erratic on an infected leaf.\n"
        f"* In the green-background images, the white lines form a consistent pattern around healthy leaves, but are more irregular and dispersed on infected leaves."
    )

