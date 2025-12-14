# Import libraries

import streamlit as st
import torch
from diffusers import StableDiffusionPipeline

# -----------------------------
# Device configuration
# -----------------------------
if torch.cuda.is_available():
    device = "cuda"
    dtype = torch.float16
else:
    device = "cpu"
    dtype = torch.float32

# -----------------------------
# Style dictionary
# -----------------------------
style_dict = {
    "none": "",
    "anime": "cartoon, animated, Studio Ghibli style, cute, Japanese animation",
    "photo": "photograph, film, 35 mm camera",
    "video game": "rendered in unreal engine, hyper-realistic, volumetric lighting",
    "watercolor": "painting, watercolors, pastel, soft composition",
}

# -----------------------------
# Load model
# -----------------------------
@st.cache_resource

def load_model():
    """
    Loads a CPU-friendly Stable Diffusion model and applies
    optimizations for inference without a GPU.
    """
    pipe = StableDiffusionPipeline.from_pretrained(
        "stabilityai/sd-turbo",
        torch_dtype=dtype,
    )

    pipe = pipe.to(device)

    # CPU optimizations
    pipe.enable_attention_slicing()

    return pipe


# -----------------------------
# Image generation function
# -----------------------------

def generate_image(prompt, pipeline, guidance, steps, style):
    full_prompt = prompt + " " + style_dict.get(style, "")

    result = pipeline(
        full_prompt,
        num_inference_steps=steps,
        guidance_scale=guidance,
        height=384,
        width=384,
    )

    return result.images[0]


# -----------------------------
# Streamlit UI
# -----------------------------

def main():
    st.set_page_config(page_title="Stable Diffusion (CPU Optimized)")
    st.title("Stable Diffusion – CPU Optimized")

    st.sidebar.header("Generation Settings")

    prompt = st.sidebar.text_area("Text-to-Image Prompt")

    guidance = st.sidebar.slider(
        "Guidance Scale",
        min_value=2.0,
        max_value=7.0,
        value=5.0,
        help="Lower values are faster and more CPU-friendly",
    )

    steps = st.sidebar.slider(
        "Inference Steps",
        min_value=5,
        max_value=30,
        value=15,
        help="More steps = better quality but much slower on CPU",
    )

    style = st.sidebar.selectbox("Style", options=list(style_dict.keys()))

    generate = st.sidebar.button("Generate Image")

    if generate:
        if not prompt.strip():
            st.warning("Please enter a prompt.")
            return

        with st.spinner("Generating image (CPU, please wait)..."):
            pipeline = load_model()
            image = generate_image(
                prompt=prompt,
                pipeline=pipeline,
                guidance=guidance,
                steps=steps,
                style=style,
            )

        st.image(image, caption="Generated Image", use_column_width=True)


if __name__ == "__main__":
    main()
