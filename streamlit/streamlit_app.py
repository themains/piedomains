import streamlit as st
from piedomains import DomainClassifier
import base64


def download_file(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="results.csv">Download results</a>'
    st.markdown(href, unsafe_allow_html=True)


def app():
    # Set app title
    st.title("piedomains: predict the kind of content hosted by a domain based on domain name and content")

    # Generic info.
    st.write(
        "The package infers the kind of content hosted by domain using the domain name, and the content, and screenshot from the homepage."
    )
    st.write("[Github](https://github.com/themains/piedomains)")

    # Set up the sidebar
    st.sidebar.title("Select Function")
    method_options = {
        "Text analysis only": "text",
        "Image analysis only": "image", 
        "Combined analysis": "combined"
    }
    selected_method = st.sidebar.selectbox("", list(method_options.keys()))

    # Create a form to enter the list of numbers
    with st.form("number_form"):
        lst_input = st.text_input("Enter a list of domains separated by commas (e.g. cnn.com, amazon.com)")

        # Add a submit button
        submitted = st.form_submit_button("Submit")

    if submitted:
        lst = [s.strip() for s in lst_input.split(",")]
        classifier = DomainClassifier()
        
        method = method_options[selected_method]
        
        if method == "text":
            transformed_df = classifier.classify_by_text(lst)
        elif method == "image":
            transformed_df = classifier.classify_by_images(lst)
        else:
            transformed_df = classifier.classify(lst)
            
        st.dataframe(transformed_df)
        download_file(transformed_df)


# Run the app
if __name__ == "__main__":
    app()
