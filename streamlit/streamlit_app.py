import streamlit as st
import pandas as pd
import piedomains
from piedomains import domain
import base64


# Define your sidebar options
sidebar_options = {
    'Predict with Text': domain.pred_shalla_cat_with_text,
    'Predict with Text and Image': domain.pred_shalla_cat_with_images
}

def download_file(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="results.csv">Download results</a>'
    st.markdown(href, unsafe_allow_html=True)

def app():
    # Set app title
    st.title("piedomains: predict the kind of content hosted by a domain based on domain name and content")

    # Generic info.
    st.write('The package infers the kind of content hosted by domain using the domain name, and the content, and screenshot from the homepage.')
    st.write('[Github](https://github.com/themains/piedomains)')

    # Set up the sidebar
    st.sidebar.title('Select Function')
    selected_function = st.sidebar.selectbox('', list(sidebar_options.keys()))

    # Create a form to enter the list of numbers
    with st.form("number_form"):
        lst_input = st.text_input("Enter a list of domains separated by commas (e.g. google.com, yahoo.com)")

        # Add a submit button
        submitted = st.form_submit_button("Submit")

    if submitted:  
        lst = [int(s.strip()) for s in lst_input.split(',')]

    if selected_function == "Predict with Text": 
        if st.button('Run'):
            transformed_df = domain.pred_shalla_cat_with_text(lst_input)
            st.dataframe(transformed_df)
            download_file(transformed_df)

    elif selected_function == "Predict with Text and Image":
        if st.button('Run'):
            transformed_df = domain.pred_shalla_cat_with_images(st_input)
            st.dataframe(transformed_df)
            download_file(transformed_df)


# Run the app
if __name__ == "__main__":
    app()
