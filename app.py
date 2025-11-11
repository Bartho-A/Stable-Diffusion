# import needed libraries
import diffusers
import streamlit as st
import torch
from PIL import ImageDraw, ImageFont

# Set the device and `dtype` for GPUs
if torch.cuda.is_available():
    device = "cuda"
    dtype = torch.float16
else:
    device = "cpu"
    dtype = torch.float32
    
# The dictionary mapping style names to style strings
style_dict = {
    "none": "",
    "anime": "cartoon, animated, Studio Ghibli style, cute, Japanese animation",
    # A photograph on film suggests an artistic approach
    "photo": "photograph, film, 35 mm camera",
    "video game": "rendered in unreal engine, hyper-realistic, volumetric lighting, --ar 9:16 --hd --q 2",
    "watercolor": "painting, watercolors, pastel, composition",
}

# Cache the model so it loads only once
@st.cache_resource
# Load Stable Diffusion (load_model function)
def load_model():
    pipeline = diffusers.StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        torch_dtype=dtype
    )

    pipeline.to(device)

    return pipeline


# The generate_images function
def generate_images(prompt, pipeline, n, guidance=7.5, steps=50, style="none"):
    return pipeline([(prompt + " " + style_dict.get(style, "")) * n], guidance_scale=guidance, num_inference_steps=steps).images


# The main function
def main():
    st.title("Stable Diffusion GUI")

    num_images = st.sidebar.number_input("Number of Images", min_value=1, max_value=10)
    prompt = st.sidebar.text_area("Text-to-Image Prompt")

    guidance_help = "Lower values follow the prompt less strictly. Higher values risk distored images."
    guidance = st.sidebar.slider("Guidance", 2.0, 15.0, 7.5, help=guidance_help)

    steps_help = "More steps produces better images but takes longer."
    steps = st.sidebar.slider("Steps", 10, 150, 50, help=steps_help)

    style = st.sidebar.selectbox("Style", options=style_dict.keys())

    generate = st.sidebar.button("Generate Images")
    if generate:
        with st.spinner("Generating images..."):
            pipeline = load_model()
            images = generate_images(
                prompt, pipeline, num_images, guidance, steps, style
            )
            for im in images:
                st.image(im)


if __name__ == "__main__":
    main()
