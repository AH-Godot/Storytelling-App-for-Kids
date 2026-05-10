import streamlit as st
from transformers import pipeline
from PIL import Image

# Set up the kid-friendly title and harder
st.title("🎠 Magic Story Maker!")
st.write("Upload a fun picture, and let's create a magical adventure together!")

# Function Part
def img2text(url):
    image_to_text_model = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
    text = image_to_text_model(url)[0]["generated_text"]
    return text

# Main Part
st.set_page_config(page_title="Magic Story Maker", page_icon="🎠")
st.header("✨ Turn Your Picture into a Magical Audio Story!")
uploaded_file = st.file_uploader("Choose a fun picture...")

if uploaded_file is not None:
    # Save the uploaded image file locally 
    bytes_data = uploaded_file.getvalue()
    with open(uploaded_file.name, "wb") as file:
        file.write(bytes_data)

    st.image(uploaded_file, caption="Your Magic Picture!", use_column_width=True)

    # Stage 1: Image Processing & Captioning
    st.text('👀 Looking at your picture...')
    scenario = img2text(uploaded_file.name)
    st.write(f"**What I see:** {scenario}")

    # Stage 2: Story Generation
    st.text('✍️ Writing a magical story...')
    story_pipe = pipeline("text-generation", model="roneneldan/TinyStories-33M")
    prompt = f"Once upon a time, there was {scenario.lower().strip()}. "
    story_results = story_pipe(prompt, max_new_tokens=100, do_sample=True, temperature=0.7, return_full_text=False)
    story = prompt + story_results[0]['generated_text']
    st.write(f"**Your Story:** {story}")

    # Stage 3: Text-to-Speech Conversion (Convert the generated text into an audio format)
    st.text('🎙️ Getting the storyteller ready...')
    audio_pipe = pipeline("text-to-speech", model="kakao-enterprise/vits-ljs")
    audio_data = audio_pipe(story)

    # Provide the audio player directly. 
    st.write("▶️ **Listen to your story:**")
    audio_array = audio_data["audio"]
    sample_rate = audio_data["sampling_rate"]
    st.audio(audio_array, sample_rate=sample_rate)
