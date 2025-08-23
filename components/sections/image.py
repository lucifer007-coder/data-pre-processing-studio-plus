import streamlit as st
from utils.data_utils import _arrowize, sample_for_preview
from preprocessing.steps import resize_image, normalize_image
from PIL import Image
import base64
import io
import logging
import pandas as pd

logger = logging.getLogger(__name__)

def section_image():
    st.header("🖼️ Image Preprocessing")
    df = st.session_state.df
    if df is None:
        st.warning("Upload a dataset first.")
        return

    try:
        # Filter columns that likely contain image paths or base64 strings
        image_cols = [
            col for col in df.columns
            if df[col].astype(str).str.contains(r'\.(png|jpg|jpeg)$|data:image', regex=True, na=False).any()
        ]
        if not image_cols:
            st.info("No columns with image paths or base64 strings detected.")
            return

        # Warning for large datasets or high-resolution images
        if len(df) > 1000:
            st.warning(
                "Large dataset detected (>1,000 rows). Image processing may be slow in a browser environment. "
                "Previews will be sampled to reduce memory usage."
            )

        with st.expander("Image Resizing", expanded=True):
            st.markdown("**Resize images to a uniform size.**")
            image_col = st.selectbox(
                "Image column",
                image_cols,
                help="Select a column with image paths or base64 strings."
            )
            width = st.number_input(
                "Width",
                16, 512, 224, 16,
                help="Target width in pixels (16-512 to balance quality and performance)."
            )
            height = st.number_input(
                "Height",
                16, 512, 224, 16,
                help="Target height in pixels (16-512 to balance quality and performance)."
            )
            if width * height > 256 * 256:
                st.warning("High-resolution images may increase memory usage. Consider lower dimensions for faster processing.")

        with st.expander("Image Normalization", expanded=True):
            st.markdown("**Normalize pixel values for machine learning compatibility.**")
            normalize = st.checkbox(
                "Normalize pixel values",
                help="Scale pixel values to [0,1] for consistency."
            )

        c1, c2, c3 = st.columns([1, 1, 1])
        with c1:
            if st.button("🔍 Preview Image Processing", help="Preview image resizing and normalization"):
                if not image_col:
                    st.warning("Please select an image column.")
                    return
                # Validate image data
                valid_images = df[image_col].astype(str).str.contains(r'\.(png|jpg|jpeg)$|data:image', regex=True, na=False)
                if not valid_images.any():
                    st.error(f"No valid image paths or base64 strings in {image_col}.")
                    return
                with st.spinner("Generating image processing preview..."):
                    prev = sample_for_preview(df, n=100)  # Limit to 100 rows for image processing
                    preview_df, resize_msg = resize_image(prev, image_col, width, height, preview=True)
                    preview_df, norm_msg = normalize_image(preview_df, image_col, preview=True) if normalize else (preview_df, "")
                    st.session_state.last_preview = (preview_df, f"{resize_msg}\n{norm_msg}")
                    st.info(f"{resize_msg}\n{norm_msg}")
                    st.dataframe(_arrowize(preview_df.head(10)))
                    # Display up to 3 sample images
                    st.subheader("Sample Processed Images")
                    valid_images = preview_df[image_col].astype(str).str.contains(r'data:image', regex=True, na=False)
                    sample_images = preview_df[valid_images][image_col].head(3)
                    cols = st.columns(min(len(sample_images), 3))
                    for idx, (col, img_data) in enumerate(zip(cols, sample_images)):
                        with col:
                            st.image(img_data, caption=f"Sample Image {idx + 1}")
        with c2:
            if st.button("📦 Add to Pipeline", help="Add image processing steps to pipeline"):
                if not image_col:
                    st.warning("Please select an image column.")
                    return
                # Validate image data before adding to pipeline
                valid_images = df[image_col].astype(str).str.contains(r'\.(png|jpg|jpeg)$|data:image', regex=True, na=False)
                if not valid_images.any():
                    st.error(f"No valid image paths or base64 strings in {image_col}.")
                    return
                steps = [
                    {"kind": "resize_image", "params": {"column": image_col, "width": width, "height": height}}
                ]
                if normalize:
                    steps.append({"kind": "normalize_image", "params": {"column": image_col}})
                st.session_state.pipeline.extend(steps)
                st.success("Added image processing steps to pipeline.")
        with c3:
            if st.button("🔄 Reset Selection", help="Clear all selections"):
                st.rerun()
    except Exception as e:
        logger.error(f"Error in section_image: {e}")
        st.error(f"Error in image section: {e}")