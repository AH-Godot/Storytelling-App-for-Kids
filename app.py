import streamlit as st
from transformers import pipeline
from PIL import Image

# Set up the kid-friendly UI
st.set_page_config(page_title="Magic Story Maker", page_icon="🪄", layout="wide")
page_bg_css = """
<style>
[data-testid="stAppViewContainer"] {
    background-color: #f0f8ff;
    background-image: radial-gradient(circle at 50% 0%, #ffdfeb 0%, #e0f7fa 50%, #fff0cc 100%);
}
[data-testid="stHeader"] {
    background-color: rgba(0,0,0,0); /* Make the top header bar transparent */
}
/* Optional: Add a slight white background to the main text area to ensure readability */
.stMarkdown {
    text-shadow: 1px 1px 2px rgba(255,255,255,0.8);
}
</style>
"""
st.markdown(page_bg_css, unsafe_allow_html=True)
st.title("🪄 Magic Story Maker! ✨")
st.write("Upload a fun picture, and let's create a magical adventure together! 🦄")
st.header("Turn Your Picture into a Story! 📖")
col1, col2 = st.columns([1, 1.2])

# Function Part 
def img2text(url):
    image2text = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
    text = image2text(url)[0]["generated_text"]
    return text

# Main Part
uploaded_file = st.file_uploader("🖼️ Choose a magical picture...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Save the uploaded image file locally to allow the models to process it
    bytes_data = uploaded_file.getvalue()
    with open(uploaded_file.name, "wb") as file:
        file.write(bytes_data)

    with col1:
        # Display the uploaded image neatly in the left column
        st.image(uploaded_file, caption="✨ Your Magic Picture! ✨", use_column_width=True)

    with col2:
        # Stage 1: Image Processing & Captioning
        with st.spinner('👀 Looking closely at your picture... 🕵️‍♂️'):
            scenario = img2text(uploaded_file.name)
        with st.expander("🔍 Peek behind the magic curtain 🧙‍♀️"):
            st.write(scenario)

        # Stage 2: Story Generation
        with st.spinner('✍️ Writing a magical story just for you... 🧚‍♂️'):
            story_pipe = pipeline("text-generation", model="roneneldan/TinyStories-33M")
            prompt = f"Once upon a time, there was {scenario.lower().strip()}. "
            story_results = story_pipe(prompt, max_new_tokens=100, do_sample=True, temperature=0.7, return_full_text=False)
            story = prompt + story_results[0]['generated_text']
        st.success("🎉 **Here is your story!** 🏰")
        st.write(story)

        # Stage 3: Text-to-Speech Conversion
        with st.spinner('🎙️ Waking up the wise old storyteller... 🦉'):
            audio_pipe = pipeline("text-to-speech", model="kakao-enterprise/vits-ljs")
            audio_data = audio_pipe(story)

        # Visual divider to separate the text from the audio player.
        st.divider()
        st.write("🎧 **Listen to your adventure:** 🎶")
        audio_array = audio_data["audio"]
        sample_rate = audio_data["sampling_rate"]
        st.audio(audio_array, sample_rate=sample_rate)

        # Trigger celebratory balloons when the whole process is successfully finished!
        st.balloons()
