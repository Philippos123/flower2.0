import streamlit as st
import matplotlib.pyplot as plt


def page_summary_body():

    st.write("### Quick Project Summary")

    st.info(
        f"#### General Information\n"
        f"Powdery mildew is a fungal disease that affects a wide range of plants. Powdery mildew diseases are caused by many different species of ascomycete fungi in the order Erysiphales.\n"
         f"#### Sampels\n"
         f"* A picture has been taken of leaf's that are either healthy or infeced with powdery mildew"
         f" Project Dataset"
         f" The dataset is taken from Kaggle wich include 2104 files for healthy leaf's and 2104 infeced with mildew")

    st.write(
        f"* For additional information, please visit and **read** the "
        f"[Project README file](https://github.com/Code-Institute-Solutions/WalkthroughProject01/blob/main/README.md).")
    

    st.success(
        f"The project has 2 business requirements:\n"
        f"* 1 - The client is interested in conducting a study to visually differentiate a cherry leaf that is healthy from one that contains powdery mildew.\n"
        f"* 2 - The client is interested in predicting if a cherry leaf is healthy or contains powdery mildew. "
        )