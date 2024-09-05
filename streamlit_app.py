import streamlit as st
import os
import pandas as pd
import base64
from PIL import Image
import io
import openai 
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import json
import re
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter, PDFPageAggregator
from pdfminer.layout import LAParams, LTImage
from pdfminer.pdfpage import PDFPage
from langchain_core.prompts import PromptTemplate
from pdf2image import convert_from_bytes
from pdfminer.layout import LTTextContainer, LTFigure, LTImage
import colorsys
import pytesseract
from PIL import Image, ImageDraw
import cv2
import numpy as np
import cv2
import numpy as np
from PIL import Image
import random
import cv2
import numpy as np

import pandas as pd
from fuzzywuzzy import fuzz


# Initialize OpenAI client
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")



def extract_text_and_images_from_pdf(file):
    resource_manager = PDFResourceManager()
    pages_content = []
    
    laparams = LAParams()
    device = PDFPageAggregator(resource_manager, laparams=laparams)
    interpreter = PDFPageInterpreter(resource_manager, device)
    
    with io.BytesIO(file.read()) as fh:
        for page in PDFPage.get_pages(fh, caching=True, check_extractable=True):
            interpreter.process_page(page)
            layout = device.get_result()
            
            text = ""
            images = []
            
            for element in layout:
                if isinstance(element, LTTextContainer):
                    text += element.get_text()
                elif isinstance(element, LTFigure):
                    # Handle figures (which may contain images)
                    for child in element:
                        if isinstance(child, LTImage):
                            images.append(child)
                elif isinstance(element, LTImage):
                    images.append(element)
            
            pages_content.append({"text": text, "images": images})
    
    return pages_content

def preprocess_text(text):
    text = re.sub(r'\s+', ' ', text).strip()
    text = text.lower()
    return text

def generate_invoice_details(invoice):
    processed_text = preprocess_text(invoice)
    
    llm = ChatOpenAI(temperature=0.7)
    
    prompt_template = """
Analyze the following invoice text and extract all relevant information.

Ensure accuracy in reading and providing values. Maintain upper case formatting for text that is in upper case. Identify and extract the required information intelligently.

Ensure all monetary values are expressed as numbers without currency symbols. Provide the most accurate and complete extraction possible based on the available information in the invoice text.

Return the extracted information as a Python dictionary with appropriate keys and values, suitable for direct conversion into a pandas DataFrame.

Invoice Text:

{invoice_text}

"""

    prompt = PromptTemplate(
        input_variables=["invoice_text"],
        template=prompt_template)
    
    chain = prompt | llm
    
    response = chain.invoke(processed_text)
    return response.content

def json_to_dataframe(json_string):
    data = json.loads(json_string)
    normalized_data = pd.json_normalize(data, sep='_')
    df = pd.DataFrame(normalized_data)
    return df

def encode_image(image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

import json
import re

def analyze_invoice_image(image, extracted_text):
    base64_image = encode_image(image)
    
    prompt_text = """You are an expert in Invoice analysis and optical character recognition (OCR).
    Analyze the provided invoice image and extracted text. Extract all relevant details in a structured JSON format.
    Pay close attention to accuracy and do not fill in missing information. If a piece of information is not clearly present in the image or text, use null.

    Validate all extracted information against both the image and the extracted text. Ensure numeric values are represented as numbers, not strings.
    If there's a discrepancy between the image and text, prioritize the information from the text.

    Extracted text for reference:
    {extracted_text}
    
    here is example of fromat you need to output json
     Extracted text for reference:
    {extracted_text}
    
    Return the extracted information as a JSON object with the following structure:

    {{
      "invoice": {{
        "seller": {{
          "name": null,
          "address": null,
          "tel": null,
          "fax": null,
          "email": null
        }},
        "buyer": {{
          "name": null,
          "address": null,
          "tel": null
        }},
        "invoice_no": null,
        "date": null,
        "currency": null,
        "items": [
          {{
            "code": null,
            "description": null,
            "quantity": null,
            "unit_price": null,
            "amount": null
          }}
        ],
        "total_amount": null,
        "total_amount_in_words": null
      }}
    }}
    change date format to "dd-mm-yyyy"
    include currency as invoice currency to "invoice"
    Do not copy values from this example. Use it only as a guide for the structure and data types.
    Provide null for any fields where you cannot find accurate information in the invoice.
    Do not include any text before or after the JSON object in your response.
    """
    
    
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text.format(extracted_text=extracted_text)},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}"
                        }
                    }
                ],
            }
        ],
        max_tokens=1000
    )

    try:
        # Try to parse the response as JSON
        return json.loads(response.choices[0].message.content)
    except json.JSONDecodeError:
        # If parsing fails, try to extract JSON from the response
        content = response.choices[0].message.content
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        
        # If all else fails, return an error message
        return {"error": "Failed to parse response as JSON", "raw_response": content}

import pandas as pd

def combine_image_reports(image_reports):
    combined_data = []
    for report in image_reports:
        if 'error' in report:
            continue

        invoice_info = report.get('invoice', {})
        items = invoice_info.get('items', [])

        # Create invoice-level information
        invoice_number = invoice_info.get('invoice_no', '')
        invoice_date = invoice_info.get('date', '')
        invoice_currency = invoice_info.get('currency', '')

        for item in items:
            row = {
                'Inv No': invoice_number,
                'Date': invoice_date,
                'Product Desc': item.get('description', ''),
                'Quantity': item.get('quantity', ''),
                'Currency': invoice_currency,
                'Rate': item.get('unit_price', ''),
                'Amount': item.get('amount', '')
            }
            combined_data.append(row)

    df = pd.DataFrame(combined_data)

    # Forward fill the invoice-level information
    df['Inv No'] = df['Inv No'].ffill()
    df['Date'] = df['Date'].ffill()
    df['Currency'] = df['Currency'].ffill()

    return df


def match_with_master_csv(generated_df, master_df):
    # Load the master CSV
   
    
    # Function to find the best match
    def find_best_match(product_desc):
        best_match = None
        best_score = 0
        for master_product in master_df['Product Desc']:
            score = fuzz.partial_ratio(product_desc.lower(), master_product.lower())
            if score > best_score:
                best_score = score
                best_match = master_product
        return best_match if best_score >= 60 else None  # 60% threshold for a match

    # Find matches and create a new DataFrame
    matched_rows = []
    for _, row in generated_df.iterrows():
        best_match = find_best_match(row['Product Desc'])
        if best_match is not None:
            master_row = master_df[master_df['Product Desc'] == best_match].iloc[0]
            new_row = {}
            for col in master_df.columns:
                if col == 'Product Desc':
                    new_row[col] = master_row[col]  # Keep the original description from master CSV
                elif col in ['Inv No', 'Date', 'Quantity', 'Currency', 'Rate']:
                    new_row[col] = row.get(col, '')
                else:
                    new_row[col] = master_row[col]
            matched_rows.append(new_row)

    # Create DataFrame with master CSV column order
    matched_df = pd.DataFrame(matched_rows, columns=master_df.columns)
    
    return matched_df



def clean_dataframe(df):
    # Remove any columns that are entirely NaN
    df = df.dropna(axis=1, how='all')
    
    # Replace NaN with empty string
    df = df.fillna('')
    
    # Trim whitespace from string columns
    for col in df.select_dtypes(include=['object']):
        df[col] = df[col].str.strip()
    
    return df


def generate_distinct_colors(n):
    colors = []
    for i in range(n):
        hue = i / n
        saturation = 0.9
        value = 0.9
        rgb = tuple(int(x * 255) for x in colorsys.hsv_to_rgb(hue, saturation, value))
        colors.append(rgb)
    return colors

def highlight_products(image, product_descriptions):
    # Convert PIL Image to OpenCV format
    opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Convert to grayscale
    gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
    
    # Apply thresholding to preprocess the image
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Perform text detection using morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    dilate = cv2.dilate(thresh, kernel, iterations=13)
    cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    # Generate a list of distinct colors
    num_colors = 10  # You can adjust this number
    colors = generate_distinct_colors(num_colors)
    # Create a copy of the image for highlighting
    highlighted_image = opencv_image.copy()

    # Loop through contours and filter for potential text regions
    for i, c in enumerate(cnts):
        area = cv2.contourArea(c)
        x,y,w,h = cv2.boundingRect(c)
        
        if area > 10 and w > 10 and h > 10:
            color = colors[i % len(colors)]
            # Draw a green rectangle around the potential text region
            cv2.rectangle(highlighted_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Convert back to PIL Image
    highlighted_pil = Image.fromarray(cv2.cvtColor(highlighted_image, cv2.COLOR_BGR2RGB))
    
    return highlighted_pil






# ==========================================================\






def main():
    st.set_page_config(layout="wide")
    st.title("Invoice Analyzer")
    st.sidebar.success("Your success message here")
    master_file = st.file_uploader(label="Enter Your Master File", type="csv", key="master")
    if master_file:
        master_csv = pd.read_csv(master_file)
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf", key="invoice")
    
    if uploaded_file is not None:
        with st.spinner(text="Loading PDF...."):    
            pages_content = extract_text_and_images_from_pdf(uploaded_file)
            all_image_reports = []
        
        with st.spinner("Extracting text"):
            pdf_images = convert_from_bytes(uploaded_file.getvalue())
            
            for i, (page, img) in enumerate(zip(pages_content, pdf_images)):
                st.subheader(f"Page {i+1}")
                
                col1, col2 = st.columns(2)
                
                with col2:
                    with st.spinner("Reading Invoices...."):
                        image_analysis = analyze_invoice_image(img, page["text"])
                        with st.expander("Generated Results"):
                            st.write(f"Image Analysis Result - Page {i+1}", image_analysis)
                        all_image_reports.append(image_analysis)
                
                if 'error' not in image_analysis:
                    st.subheader(f"DataFrame for Page {i+1}")
                    df = combine_image_reports([image_analysis])
                    st.dataframe(df)
                
                with col1:
                    if 'error' not in image_analysis:
                        invoice_data = image_analysis['invoice']
                        product_descriptions = [item['description'] for item in invoice_data['items'] if item['description']]
                        
                        highlighted_img = highlight_products(img, product_descriptions)
                        st.image(highlighted_img, caption=f"Highlighted Product Descriptions - Page {i+1}", use_column_width=True)
                    else:
                        st.error(f"Error in image analysis for Page {i+1}: {image_analysis['error']}")
        
        with st.spinner("Generating Reports"):
            if all_image_reports:
                st.subheader("Combined DataFrame of All Pages")
                combined_df = combine_image_reports(all_image_reports)
                st.dataframe(combined_df)
                
                st.subheader("Matched DataFrame with Master CSV")
                matched_df = match_with_master_csv(combined_df, master_csv)
                st.dataframe(matched_df, height=300)
                
                # Option to download the combined CSV
                csv = matched_df.to_csv(index=False)
                st.download_button(
                    label="Download Combined CSV",
                    data=csv,
                    file_name="Esskay.csv",
                    mime="text/csv",
                )

if __name__ == "__main__":
    main()