import streamlit as st
from diffusers import StableDiffusionPipeline
from transformers import pipeline
from gtts import gTTS
import os
import torch
torch.cuda.empty_cache()
import tempfile
from PIL import Image
from io import BytesIO

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Load Stable Diffusion model
model_id = "CompVis/stable-diffusion-v1-4"
sd_pipeline = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, revision="fp16",use_auth_token=True) #ENTER YOUR API TOKEN FROM HUGGING FACE HERE INSTEAD OF TRUE

# Enable memory-efficient attention slicing
sd_pipeline.enable_attention_slicing()

try:
    sd_pipeline.enable_xformers_memory_efficient_attention()
except Exception as e:
    print("xFormers memory efficient attention not available:", e)

sd_pipeline.to("cuda")

# Load text generation model
hf_text_gen = pipeline("text-generation", model="gpt2", max_length=150)

def text2image(text):
    image = sd_pipeline(text).images[0]
    return image

def generate_story(scenario):
    prompt = (
        "You are a story teller. You can generate a short story based on a simple narrative, "
        "and the story should be no more than 50 words.\n\n"
        f"CONTEXT: {scenario}\n"
        "STORY: "
    )
    story_output = hf_text_gen(prompt, do_sample=True, truncation=True, temperature=1.0)[0]['generated_text']
    story = story_output[len(prompt):].strip()
    return story

def text_to_audio(text, lang='en'):
    tts = gTTS(text=text, lang=lang)
    temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(temp_audio.name)
    return temp_audio.name

# Streamlit UI
st.title("AI Story & Image Generator")
st.write("Enter a narrative below, and I'll generate an image, a short story, and an audio narration!")

user_input = st.text_area("Enter your narrative:")

if st.button("Generate"):
    if user_input:
        with st.spinner("Generating image..."):
            generated_image = text2image(user_input)
            img_buffer = BytesIO()
            generated_image.save(img_buffer, format="PNG")
            img_buffer.seek(0)
            st.image(img_buffer, caption="Generated Image", use_column_width=True)

        with st.spinner("Generating story..."):
            generated_story = generate_story(user_input)
            st.subheader("Generated Story")
            st.write(generated_story)

        with st.spinner("Converting story to audio..."):
            audio_file = text_to_audio(generated_story)
            st.audio(audio_file, format="audio/mp3")
    else:
        st.warning("Please enter a narrative!")
