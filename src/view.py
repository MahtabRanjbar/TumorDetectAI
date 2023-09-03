import streamlit as st


class TumorDetectionView:
    def __init__(self):
        # Set the title of the app
        st.set_page_config(page_title="TumorDetectAI", page_icon="microscope")

        # Define some custom CSS
        st.markdown(
            """
            <style>
            .file_input {
                background-color: rgba(255, 255, 255, 0.7);
                border-radius: 10px;
                padding: 20px;
                margin-bottom: 20px;
            }
            .file_input .label {
                font-size: 20px !important;
            }
            .file_input .content {
                padding-top: 15px !important;
            }
            .submit_button {
                background-color: #4CAF50;
                color: white;
                padding: 10px 20px;
                border: none;
                border-radius: 4px;
                cursor: pointer;
            }
            .submit_button:hover {
                background-color: #3e8e41;
            }
            .prediction-box {
                border: 2px solid #ccc;
                border-radius: 10px;
                padding: 20px;
                margin-top: 20px;
                background-color: #f1f1f1;
                box-shadow: 5px 5px 5px #ccc;
            }
            .prediction-label {
                color: black;
                font-size: 24px;
                font-weight: bold;
                margin-bottom: 10px;
            }

            .prediction-result {
                color: black;
                font-size: 18px;
                font-weight: bold;
            }
            .description {
                background-color:#F0FFF0;
                border-radius: 10px;
                padding: 20px;
                margin-bottom: 20px;
                box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.2);
            }
            .description h1 {
                 color: black;
                font-size: 40px;
                font-weight: bold;
                text-align: center;
                margin-bottom: 30px;
            }
            .description p {
                color: black;
                font-size: 20px;
                text-align: justify;
                line-height: 1.5;
                margin-bottom: 20px;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

    def get_uploaded_file(self):
        uploaded_file = st.file_uploader(
            "Choose a Brain MRI image:",
            type=["jpeg", "jpg", "png"],
            key="file_uploader",
        )
        return uploaded_file

    def show_prediction(self, image, label):
        st.image(image, caption="Uploaded Image", use_column_width=True, width=300)
        st.markdown(
            """
            <div class="prediction" style='background-color: #3CB371; border: 2px solid #e6e6e6; border-radius: 5px; padding: 5px; margin-top: 20px; display: inline-block;'>
                <p class=prediction-label >Prediction:</p>
                <p class=prediction-result >{}</p>
            </div>
            """.format(
                label
            ),
            unsafe_allow_html=True,
        )

    def show_description(self):
        st.markdown(
            """
            <div class="description">
            <h1>Brain Tumor Detection App</h1>
            <p>TumorDetectAI is a web application that utilizes artificial intelligence to detect brain tumors in MRI images. With the help of advanced deep learning algorithms, the application offers accurate and efficient predictions regarding the presence of brain tumors. Users can conveniently upload their brain MRI images and receive instant results.
</p>
            <p>The model used in this app was trained on a dataset ofBrain MRI images from patients with and without Brain tumor. The model achieved an accuracy of about 90% on a held-out test set, indicating that it is a reliable tool for detecting Brain Tumor from Brain MRI images.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
