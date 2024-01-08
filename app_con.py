import io
import os
import re
import fitz
import base64
import requests
from PIL import Image
import streamlit as st

import pandas as pd
import numpy as np
import json



def pdf_to_images(uploaded_file):
    images = []

    # Save the uploaded file to a temporary location
    with open("temp.pdf", "wb") as temp_file:
        temp_file.write(uploaded_file.getvalue())

    pdf_document = fitz.open("temp.pdf")

    for page_number in range(pdf_document.page_count):
        page = pdf_document[page_number]
        pixmap = page.get_pixmap()
        image = Image.frombytes("RGB", [pixmap.width, pixmap.height], pixmap.samples)
        images.append(image)

    pdf_document.close()

    # Remove the temporary PDF file
    os.remove("temp.pdf")

    return images


def encode_image(image):
    with io.BytesIO() as buffer:
        image.save(buffer, format="JPEG")
        return base64.b64encode(buffer.getvalue()).decode('utf-8')


@st.cache_data
def completion_api_single(prompt, base64_image):
    payload = {
        "model": "gpt-4-vision-preview",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 4096
    }

    return payload


def convert_data():
    api_key = os.getenv('OPENAI_API_KEY')
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

    prompt = '''
    Image Text Extraction:
    Confirm that all text content, including headers and footers, is successfully captured.

    JSON Format:
    Organize the extracted text into a JSON format.
    Each key in the JSON should correspond to a specific type of information (e.g., Document Title, Date, Currency, etc.).

    Document Title:
    Identify and extract the document title from the text.
    Assign this to the corresponding key in the JSON.

    Format Inconsistency:
    Check for any inconsistencies in formatting throughout the document.
    If there are variations, create a mapping system to standardize the format under each relevant key in the JSON.

    Standard Currency:
    Look for currency information in the text.
    Ensure that all currency values are converted to a standard currency format.

    Date Time Formatting:
    Identify date and time information in the text.
    Standardize the date format to dd/mm/yyyy.

    Invoice Placement:
    Segment the JSON output to clearly distinguish the invoice-related information.
    The JSON structure should be organized to clearly represent each segment of the invoice (e.g., header, line items, totals).

    Consistency Across the Document:
    JSON format will be used later to convert this information into a pandas dataframe so ensure the output is consistent and compatible with this format.
    Confirm that the JSON format is consistent throughout the entire document.
    Ensure that the same keys are used for the same types of information.

    Quality Check:
    Perform a final check to ensure accuracy and completeness.
    Verify that all relevant information is captured, standardized, and organized in the JSON output.
    '''

    status = None
    content = None

    if uploaded_file is not None:

        st.write("### PDF Preview")

        pdf_images = pdf_to_images(uploaded_file)

        img = None

        with st.status("Processing PDF to JSON..."):
            st.write("Reading PDF...")
            for idx, image in enumerate(pdf_images):
                img = image
                st.image(image, caption=f"Page {idx + 1}", use_column_width=True)

            st.write("Extracting PDF data...")
            base64_image = encode_image(img)
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            }

            st.write("Converting Data to JSON...")
            payload = completion_api_single(prompt, base64_image)

            with st.spinner('Wait for it...'):
                response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

            st.write("Finishing...")
            status = False
            try:
                content = response.json()['choices'][0]['message']['content']
                status = True
            except KeyError:
                st.write(content)
                st.write("Sorry it seems we ran into some issues, can you try again please")

    return status, content

def display_content(content):
  st.divider()

  with st.expander("JSON Output"):
      st.write("```json\n" + content + "\n```")
      
  with st.expander("Table Output"):
      pattern = r'"([^"]+)":\s*"([^"]+)"'
      matches = re.findall(pattern, content)
      data_dict = dict(matches)
      df = pd.DataFrame(list(data_dict.items()), columns=['Key', 'Value'])

      st.table(df) 



def main():
  st.set_page_config(page_title='LLM Automation', layout='wide')

  st.image("./CBL_Logo.png")

  st.title("LLM aided Invoice Data Extraction")
  st.write("### Upload a PDF file to extract data")

  _ , content = convert_data()
  if content is not None:
    display_content(content)


if __name__ == "__main__":
    main()
