import streamlit as st

def display_help_content():
    st.header("🆘 Help & Instructions")
    
    st.subheader("How to use this application")
    st.markdown("""
    This application offers two ways to check your heart rate:
    
    1. **Live Webcam Check**:
       * Select the "Check Webcam" tab
       * Click the "Start Webcam" button
       * Position your face clearly in view of the camera
       * Stay still for more accurate readings
       
    2. **Video Upload**:
       * Select the "Upload Video" tab
       * Upload a video file of a person's face
       * The app will process the video and estimate heart rate
       * Results can be downloaded as CSV
    """)
    
    st.subheader("Tips for accurate readings")
    st.markdown("""
    * Ensure good, even lighting on your face
    * Face the camera directly
    * Minimize movement during measurement
    * For videos, use high quality recordings with clear facial visibility
    """)
    
    st.subheader("Troubleshooting")
    st.markdown("""
    * **No faces detected**: Make sure your face is clearly visible and well lit
    * **Inaccurate readings**: Try improving lighting conditions and minimize movement
    * **Video processing issues**: Ensure your video contains a clearly visible face
    """)