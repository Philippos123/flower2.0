import streamlit as st
import matplotlib.pyplot as plt
import os


def page_project_hypothesis_body():
    st.write("### Project Hypothesis and Validation")

    st.success(
        f"* We suspect that infected leafs and healthy leafs have clear marks/signs. \n\n"
        f"* An image montage belov will show you what we found out about after the machine run through the dataset.\n"
        f"* Belov the image montage we will also leave an explenation about what we found out and the clear signs we found")



    st.write("#### Here is an analysis of average healthy leaf")

    # Use the relative path, ensure the image is in the correct location
    st.image('outputs/v1/avg_var_healthy.png', caption="Average healthy leaf\n")

    st.write("#### Here is an analysis of average Powder mildew infected leaf")

    # Use the relative path, ensure the image is in the correct location
    st.image('outputs/v1/avg_var_powdery_mildew.png', caption="Average infected leaf\n\n")

    st.success(
        f" #### As we can see there is a diffrence between the healthy and infected leafs.\n"
        f"* You can see it from the white lines on the black pictures\n"
        f"* You can also see it in the green pictures that the white lines goes around the leaf on the healthy leaf while on the infected leaf the lines are more unpredictible."
    )