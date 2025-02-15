import streamlit as st
from diffusers import StableDiffusionPipeline  # For generating images from text prompts
from transformers import pipeline  # For text generation using GPT-2
from gtts import gTTS  # For text-to-speech conversion
import os
import torch
torch.cuda.empty_cache()  # Free up GPU memory
import tempfile  # For handling temporary files
from PIL import Image  # Image processing
from io import BytesIO  # For handling image data in memory

# Optimize CUDA memory allocation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Load the Stable Diffusion model for text-to-image generation
model_id = "CompVis/stable-diffusion-v1-4"
sd_pipeline = StableDiffusionPipeline.from_pretrained(
    model_id, torch_dtype=torch.float16, revision="fp16", use_auth_token=True  # ENTER YOUR API TOKEN FROM HUGGING FACE HERE INSTEAD OF TRUE
)

# Enable memory-efficient attention slicing for better performance
sd_pipeline.enable_attention_slicing()

# Attempt to enable xFormers for further memory efficiency (if available)
try:
    sd_pipeline.enable_xformers_memory_efficient_attention()
except Exception as e:
    print("xFormers memory efficient attention not available:", e)

# Move the model to GPU for faster processing
sd_pipeline.to("cuda")

# Load the text generation model (GPT-2) for generating short stories
hf_text_gen = pipeline("text-generation", model="gpt2", max_length=150)

def text2image(text):
    """Generate an image based on the given text prompt using Stable Diffusion."""
    image = sd_pipeline(text).images[0]
    return image

def generate_story(scenario):
    """Generate a short story (max 50 words) based on a given scenario."""
    prompt = (
        "You are a story teller. You can generate a short story based on a simple narrative, "
        "and the story should be no more than 50 words.\n\n"
        f"CONTEXT: {scenario}\n"
        "STORY: "
    )
    story_output = hf_text_gen(prompt, do_sample=True, truncation=True, temperature=1.0)[0]['generated_text']
    story = story_output[len(prompt):].strip()  # Extract the generated text after the prompt
    return story

def text_to_audio(text, lang='en'):
    """Convert text into speech and save it as an audio file."""
    tts = gTTS(text=text, lang=lang)
    temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")  # Create a temporary audio file
    tts.save(temp_audio.name)  # Save the speech as an MP3 file
    return temp_audio.name  # Return the file path

# Streamlit UI
st.title("AI Story & Image Generator")  # Page title
st.write("Enter a narrative below, and I'll generate an image, a short story, and an audio narration!")

# User input text box
user_input = st.text_area("Enter your narrative:")

if st.button("Generate"):
    if user_input:
        # Generate an image based on user input
        with st.spinner("Generating image..."):
            generated_image = text2image(user_input)
            img_buffer = BytesIO()
            generated_image.save(img_buffer, format="PNG")  # Save image to buffer
            img_buffer.seek(0)
            st.image(img_buffer, caption="Generated Image", use_column_width=True)

        # Generate a short story based on user input
        with st.spinner("Generating story..."):
            generated_story = generate_story(user_input)
            st.subheader("Generated Story")
            st.write(generated_story)

        # Convert the generated story into speech
        with st.spinner("Converting story to audio..."):
            audio_file = text_to_audio(generated_story)
            st.audio(audio_file, format="audio/mp3")  # Play the generated audio

    else:
        st.warning("Please enter a narrative!")  # Warn user if no input is given
