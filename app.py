import streamlit as st
from transformers import pipeline

# --- Text Generation Pipeline Setup ---
@st.cache_resource
def load_generator():
    return pipeline("text-generation", model="gpt2")

generator = load_generator()

def generate_text(prompt_text, max_length=50, num_return_sequences=1):
    try:
        # Truncation is important for longer prompts
        result = generator(prompt_text, max_length=max_length, num_return_sequences=num_return_sequences, truncation=True)
        return result[0]['generated_text']
    except Exception as e:
        return f"An error occurred during text generation: {e}"

# --- Streamlit Frontend ---
st.set_page_config(page_title="Text Generator", page_icon="✍️")

st.title("✍️ Simple Text Generator")
st.markdown("Enter a prompt below, and the AI will complete it for you!")

# Text input area
user_prompt = st.text_area(
    "Enter your starting prompt:",
    value="Ai is going to replace because",
    height=100
)

# Max length slider
max_length = st.slider("Max Generated Text Length:", min_value=10, max_value=200, value=50)

# Generate button
if st.button("Generate Text"):
    if user_prompt:
        with st.spinner(f"Generating text with max length {max_length}..."):
            generated_text = generate_text(user_prompt, max_length=max_length)
            st.subheader("Generated Text:")
            st.success(generated_text)
    else:
        st.warning("Please enter a prompt to generate text.")

st.markdown("---")
st.info("Powered by GPT-2 model from Hugging Face Transformers.")
