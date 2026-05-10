import streamlit as st
from transformers import pipeline
from PIL import Image

# Set up the app title and introductory text for a kid-friendly experience
st.title("🎠 Magic Story Maker!")
st.write("Upload a fun picture, and let's create a magical adventure together!")

# Function part
def img2text(url):
    image_to_text_model = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
    text = image_to_text_model(url)[0]["generated_text"]
    return text

# Main part
st.set_page_config(page_title="Magic Story Maker", page_icon="🎠")
st.header("✨ Turn Your Picture into a Magical Audio Story!")
uploaded_file = st.file_uploader("Choose a fun picture...")

if uploaded_file is not None:
    # Save file locally
    bytes_data = uploaded_file.getvalue()
    with open(uploaded_file.name, "wb") as file:
        file.write(bytes_data)

    st.image(uploaded_file, caption="Your Magic Picture!", use_column_width=True)

    # Stage 1: Image to Text (Using the function)
    st.text('👀 Looking at your picture...')
    scenario = img2text(uploaded_file.name)
    st.write(f"**What I see:** {scenario}")

    # Stage 2: Text to Story (Inline)
    st.text('✍️ Writing a magical story...')
    story_pipe = pipeline("text-generation", model="pranavpsv/genre-story-generator-v2")
    prompt = "Write a story for 3-10 years old kids" + scenario
    story_results = story_pipe(prompt)
    story = story_results[0]['generated_text']
    st.write(f"**Your Story:** {story}")

    # Stage 3: Story to Audio (Inline)
    st.text('🎙️ Getting the storyteller ready...')
    audio_pipe = pipeline("text-to-audio", model="Matthijs/mms-tts-eng")
    audio_data = audio_pipe(story)

    # Play button
    if st.button("▶️ Read my story!"):
        audio_array = audio_data["audio"]
        sample_rate = audio_data["sampling_rate"]
        st.audio(audio_array, sample_rate=sample_rate)
