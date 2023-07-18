#importing libraries
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import img_to_array
import streamlit as st
import cv2
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase,RTCConfiguration
import streamlit.components.v1 as components
import time
import threading


#page title
st.set_page_config(page_title="FACE-IT", page_icon="Images/logo.png")
#load model
model = tf.keras.models.load_model("emotion_detection.h5")
#face detection classifier
try:
    face_haar_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
except Exception:
    st.write("Error loading cascade classifiers")

# Global variables to store emotion counts
happy_count = 0
neutral_count = 0
angry_count = 0

# Function to update emotion counts every 5 seconds
def update_emotion_counts():
    global happy_count, neutral_count, angry_count
    while True:
        time.sleep(5)
        # The counts will be updated and displayed in the Streamlit app, no need to print here
        pass

class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        super().__init__()
        self.happy_count = 0
        self.neutral_count = 0
        self.angry_count = 0
        self.last_count_time = time.time()

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        #image gray
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #detect faces
        faces = face_haar_cascade.detectMultiScale(image=gray_image)
        for (x, y, w, h) in faces:
            cv2.rectangle(img=img, pt1=(x, y), pt2=(
                x+w, y+h), color=(255, 0, 0), thickness=2)
            roi_gray = gray_image[y-5:y+h+5, x-5:x+w+5]
            roi_gray = cv2.resize(roi_gray, (48, 48))
            image_pixels = img_to_array(roi_gray)
            image_pixels = np.expand_dims(image_pixels, axis=0)
            #normalize
            image_pixels /= 255
            predictions = model.predict(image_pixels)
            max_index = np.argmax(predictions[0])
            #map predictions
            emotion_detection = ('angry', '', '', 'happy', '', '', 'neutral')
            emotion_prediction = emotion_detection[max_index]
            font = cv2.FONT_HERSHEY_SIMPLEX
            label_color = (0, 255, 0)

             # Increment the respective emotion count
            if emotion_prediction == "happy":
                self.happy_count += 1
            elif emotion_prediction == "neutral":
                self.neutral_count += 1
            elif emotion_prediction == "angry":
                self.angry_count += 1

            # Calculate time difference since last count update
            current_time = time.time()
            time_difference = current_time - self.last_count_time



            # If 5 seconds have passed, update the emotion counts on Streamlit
            if time_difference >= 5:
                self.last_count_time = current_time
                # Clear the Streamlit output area before updating counts
                st.empty()
                # Display the emotion counts on Streamlit
                st.write("Emotion Counts:")
                st.write("Happy:", self.happy_count)
                st.write("Neutral:", self.neutral_count)
                st.write("Angry:", self.angry_count)

            # Print the emotion count to the terminal
            print("Emotion Count - Happy:", self.happy_count, "Neutral:", self.neutral_count, "Angry:", self.angry_count)
            cv2.putText(img, emotion_prediction, (int(x), int(y)), font, 0.9, label_color, 2)
            print("Detected Emotion:", emotion_prediction)  # Print the emotion to the terminal
            
        return img

def main():
    # Application
    pages = ["Home","Analysis"]
    with st.sidebar:
        st.title('Page Selection')
        page_name = st.selectbox("Select Page:", pages)
    st.title(page_name)

    if page_name == 'Home':
        home_html = """<body>
                    <h4 style="font-size:30px">Real Time Emotion Detection</h4>
                    <p>The application detects faces and predicts the face emotion using OpenCV and a customized CNN model trained on FER2013 dataset.</p>
                    </body>"""
        st.markdown(home_html,unsafe_allow_html=True)
        st.write("Click on start to use a webcam and detect your Facial Emotion.")
        webrtc_streamer(key="example", video_transformer_factory=VideoTransformer,media_stream_constraints={
            "video": True,
            "audio": False
        },rtc_configuration=RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
))

# Main function to run the Streamlit application
def main():
    # Application pages
    pages = ["Home", "Analysis"]
    
    # Sidebar page selection
    with st.sidebar:
        st.title('Page Selection')
        page_name = st.selectbox("Select Page:", pages)
    st.title(page_name)
    
    # Page content based on selection
    if page_name == 'Home':
        home_html = """<body>
                    <h4 style="font-size:30px">Real Time Emotion Detection</h4>
                    <p>The application detects faces and predicts the face emotion using OpenCV and a customized CNN model trained on FER2013 dataset.</p>
                    </body>"""
        st.markdown(home_html, unsafe_allow_html=True)
        st.write("Click on start to use a webcam and detect your Facial Emotion.")
        webrtc_streamer(key="example", video_transformer_factory=VideoTransformer, media_stream_constraints={
            "video": True,
            "audio": False
        }, rtc_configuration=RTCConfiguration(
            {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        ))

        # Create a layout using columns to position the link in the lower-left corner
        col1, col2 = st.columns([4, 1])
        col1.markdown("")
        col1.markdown(
            """
            <a href="http://localhost:8100/dashboard" target="_self" style="font-size: 16px;">Go back to Main Page</a>
            """,
            unsafe_allow_html=True
        )


    elif page_name == "Analysis":
        global happy_count, neutral_count, angry_count

        # Start the thread to update emotion counts every 5 seconds
        import threading
        update_thread = threading.Thread(target=update_emotion_counts)
        update_thread.daemon = True
        update_thread.start()

        st.title("Emotion Analysis")

        # Create a Streamlit button to manually trigger the update of emotion counts
        if st.button("Update Emotion Counts"):
            # Clear the Streamlit output area before updating counts
            st.empty()
            # Display the emotion counts on Streamlit
            st.write("Emotion Counts:")
            st.write("Happy:", video_transformer.happy_count)
            st.write("Neutral:", video_transformer.neutral_count)
            st.write("Angry:", video_transformer.angry_count)

        # Create a layout using columns to position the link in the lower-left corner
        col1, col2 = st.columns([4, 1])
        col1.markdown("")
        col1.markdown(
            """
            <a href="http://localhost:8100/dashboard" target="_self" style="font-size: 16px;">Go back to Main Page</a>
            """,
            unsafe_allow_html=True
        )

if __name__ == "__main__":
    model = tf.keras.models.load_model("emotion_detection.h5")
    try:
        face_haar_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    except Exception:
        st.write("Error loading cascade classifiers")


    # Add this CSS style to your Streamlit app
    hide_st_style = """
        <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            .stButton button {
                background-color: rgb(255, 75, 75); 
                color: white;
            }
        </style>
    """
    st.markdown(hide_st_style, unsafe_allow_html=True)

    # Create an instance of VideoTransformer
    video_transformer = VideoTransformer()

    main()