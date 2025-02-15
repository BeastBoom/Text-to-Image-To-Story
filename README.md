# AI Story and Image Generator

## Overview

The AI Story and Image Generator is an interactive AI agent that transforms your text narratives into multimedia content. It leverages state-of-the-art models to:

- **Generate Images:** Convert your narrative into an image using the Stable Diffusion model.
- **Create Stories:** Generate a creative, short story (up to 50 words) from your narrative using GPT-2.
- **Convert Text to Audio:** Transform the generated story into an audio file using Google Text-to-Speech (gTTS).
- **Provide a User-Friendly Interface:** Built with Streamlit, the app offers a simple web interface and public deployment using LocalTunnel.

## Features

- **Text-to-Image Generation:** Produce images that visualize your narrative.
- **Story Generation:** Get creative, concise stories based on your input.
- **Audio Narration:** Listen to your story via an automatically generated audio file.
- **Easy Deployment:** Run and expose the app publicly using Streamlit and LocalTunnel.

## Setup

### Prerequisites

- **Python 3.7+**
- A GPU is recommended for optimal performance (though CPU mode is available with reduced performance).
- A valid Hugging Face API token is required for accessing Stable Diffusion.

### Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/BeastBoom/Text-to-Image-To-Story

2. **Install Dependencies:**
   
   ```bash
   pip install -r requirements.txt

3. **Retrieve Your Public IP Address:**

   ```bash
   wget -q -O - ipv4.icanhazip.com
COPY THE SHOWN IP ADDRESS

Running the Application
-----------------------

1.  **Start the App and Expose It Publicly:**

    Run the following command in your terminal:

    `streamlit run app.py & npx localtunnel --port 8501`

    After executing this command, LocalTunnel will display a public URL. Click on the link provided to access the web interface of your AI agent.

2. **Paste the IP Address in the prompt opened**
![image](https://github.com/user-attachments/assets/fa2faaca-2460-4c94-9674-d39cc5473ed7)


3.  **Configure Your Hugging Face API Token:**

    In the `app.py` file, locate the following line:
    
    `sd_pipeline = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, revision="fp16", use_auth_token=True)`

    Replace `True` with your actual Hugging Face API token as a string. For example:
    ```bash
    sd_pipeline = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, revision="fp16", use_auth_token="YOUR_HF_API_TOKEN")

Running in Google Colab
-----------------------

If you prefer to run the project in Google Colab, follow these steps:

1.  **Open a New Notebook:**

    Open Google Colab and create a new notebook.

2.  **Install Dependencies:**

    In the first cell, run:
    
    ```bash
    pip install -r requirements.txt
    pip install streamlit pyngrok

4.  **Upload Your Repository Files:**

    You can either clone your repository directly in Colab:

    ```bash
    git clone https://github.com/yourusername/your-repo.git
    %cd your-repo

  Or upload your files manually.

4.  **Retrieve Your Public IP Address:**

    In a new cell, run:
    
    ```bash
    wget -q -O - ipv4.icanhazip.com

  Copy the displayed IP address.

6.  **Run the App and Expose It Publicly:**

    In last cell, run:
    ```bash
    streamlit run app.py & npx localtunnel --port 8501

  After running this command, a link will be printed. Click the last link displayed to access your deployed app.

Additional Information
----------------------

-   **Deployment:** Combining Streamlit and LocalTunnel allows for rapid public deployment of your AI agent.
-   **Performance:** For best performance, use a system with an NVIDIA GPU. Running on CPU is possible but will result in slower image generation.
-   **Customization:** Feel free to modify inference parameters (e.g., inference steps, image resolution) in `app.py` to balance speed and quality.
