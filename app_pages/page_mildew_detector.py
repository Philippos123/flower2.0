import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
from src.data_management import download_dataframe_as_csv
from src.machine_learning.predictive_analysis import (
    load_model_and_predict,
    resize_input_image,
    plot_predictions_probabilities
)


def page_mildew_detector_body():
    st.write(
        "* You can download a set of mildew infected or healthy leaf"
        "samples for live prediction. "
        "You can download the images from"
        "[here](https://www.kaggle.com/datasets/codeinstitute/cherry-leaves?resource=download)."
    )

    st.write("---")

    images_buffer = st.file_uploader(
        'Upload leaf samples. You may select more than one.',
        type=['png', 'jpg', 'jpeg'],  # Include jpg and jpeg formats
        accept_multiple_files=True
    )

    # Define label mapping for swapping
    label_mapping = {
        'Healthy': 'Infected',
        'Infected': 'Healthy'
    }

    if images_buffer:
        df_report = pd.DataFrame([])
        for image in images_buffer:
            img_pil = Image.open(image)
            st.info(f"Leaf sample: **{image.name}**")
            img_array = np.array(img_pil)
            st.image(img_pil, caption=f"Image Size: {img_array.shape[1]}px width x {img_array.shape[0]}px height")

            # Resize and preprocess the image
            resized_img = resize_input_image(img=img_pil, version='v1')
            pred_proba, pred_class = load_model_and_predict(resized_img, version='v1')

            # Swap the prediction class labels
            adjusted_pred_class = label_mapping.get(pred_class, pred_class)

            # Plot prediction probabilities
            plot_predictions_probabilities(pred_proba, adjusted_pred_class)

            # Prepare data for the report
            df_report = df_report.append({
                'File Name': image.name,
                'Prediction Probability': pred_proba,
                'Prediction Class': adjusted_pred_class
            }, ignore_index=True)

            # Display the adjusted prediction result
            st.write(f"The predictive analysis indicates the sample leaf is {adjusted_pred_class}.")

        if not df_report.empty:
            st.success("Analysis Report")
            st.table(df_report)
            st.markdown(download_dataframe_as_csv(df_report), unsafe_allow_html=True)
