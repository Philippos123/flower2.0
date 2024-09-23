import streamlit as st
import matplotlib.pyplot as plt

def page_summary_body():
    st.write("### Quick Project Summary")

    st.info(
        f"#### General Information\n"
        f"Powdery mildew is a fungal disease that affects a wide range of plants. It is caused by various species of ascomycete fungi in the order *Erysiphales*. The disease is of particular concern to the client due to its impact on their cherry plantation crops.\n"
        
        f"#### Dataset Content\n"
        f"The dataset contains 4,208 images of individual cherry leaves taken against a neutral background. The images are sourced from the client's crop fields and are classified as either healthy or infested by powdery mildew, a biotrophic fungus that poses a serious threat to cherry crops. The dataset is sourced from Kaggle.\n\n"
        
        f"**Data Distribution and Dimensions**\n"
        f"- Healthy Leaves: 1,052 images\n"
        f"- Infected Leaves: 3,156 images\n"
        f"- Training Data: 2,944 images (resized to 256x256 pixels)\n"
        f"- Images are in RGB format with 3 channels (R, G, B) and a batch size of 20.\n\n"
        
        f"#### Business Problem\n"
        f"**Client**: Farmy & Foods, an agricultural company.\n"
        f"**Goal**: Develop a machine learning system to automatically detect powdery mildew on cherry tree leaves using image-based analysis.\n"
        f"**Motivation**: The current manual disease inspection process is time-consuming and inefficient. Automating this process will enable faster, scalable inspections and timely interventions using fungicides across multiple farms.\n\n"
        
        f"#### Key Requirements\n"
        f"- **Detection**: Accurately classify whether a cherry leaf is healthy or infected by powdery mildew based on an image.\n"
        f"- **Real-time Use**: The solution must be fast enough for real-time use, potentially on mobile devices for field inspections.\n"
        f"- **Report Generation**: Generate a report based on the examined leaf images, indicating the health status of the cherry trees."
    )

    st.write(
        f"#### Example Samples\n"
        f"- Images of cherry leaves were taken in various conditions, classified as either healthy or infected with powdery mildew.\n"
    )

    st.write(
        f"* For more detailed information, you can visit the "
        f"[Project README file](https://github.com/Code-Institute-Solutions/WalkthroughProject01/blob/main/README.md)."
    )

    st.success(
        f"### Business Requirements\n"
        f"The project has two main business requirements:\n"
        f"1. **Visual Study**: The client seeks to visually differentiate healthy cherry leaves from those infected with powdery mildew.\n"
        f"2. **Prediction**: The client wants a prediction model to classify whether a cherry leaf is healthy or infected based on an image."
    )