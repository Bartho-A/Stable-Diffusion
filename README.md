# Stable Diffusion Streamlit GUI

A simple Streamlit web app to generate AI images using Stable Diffusion. Users can input a prompt, choose styles, and adjust generation parameters to create unique images.

## Features
- Text-to-image generation using Stable Diffusion
- Selectable styles: anime, photo, video game, watercolor, or none
- Adjustable guidance scale and inference steps
- Progress indicators: progress bar updates per image
- Image download button
- Batch generation with queueing
- Sidebar text area for negative prompts

## Installation

1. Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/stable-diffusion-streamlit.git
cd stable-diffusion-streamlit

## Install dependencies
pip install -r requirements.txt

## Usage
Run the Streamlit app
streamlit run app.py

## Notes
-	GPU is recommended for faster image generation.
- Style presets can be extended in style_dict inside app.py.

MIT License
Copyright (c) 2025 Bartholomeow Aobe

