import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the LSTM Model
model = load_model('next_word_lstm.h5')

# Load the tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Function to predict the next word
def predict_next_word(model, tokenizer, text, max_sequence_len):
    token_list = tokenizer.texts_to_sequences([text])[0]
    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len-1):]  # Ensure the sequence length matches max_sequence_len-1
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted, axis=1)
    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
    return None

# Streamlit app configuration
st.set_page_config(
    page_title="Next Word Prediction",
    page_icon="🔮",
    layout="wide",
)

# App title and description
st.title("🔮 Next Word Prediction With LSTM")
st.markdown(
    """
    Welcome to the **Next Word Prediction App**!  
    Enter a sequence of words or upload a text file to predict the next word using an LSTM model.
    """
)

# Sidebar for navigation
st.sidebar.header("Navigation")
options = ["Home", "Upload File", "About"]
choice = st.sidebar.radio("Go to", options)

if choice == "Home":
    st.subheader("Predict the Next Word")
    input_text = st.text_input("Enter the sequence of words", "To be or not to")
    if st.button("Predict Next Word"):
        max_sequence_len = model.input_shape[1] + 1  # Retrieve the max sequence length from the model input shape
        next_word = predict_next_word(model, tokenizer, input_text, max_sequence_len)
        st.write(f'Next word: **{next_word}**')

elif choice == "Upload File":
    st.subheader("Upload Your Text File")
    uploaded_file = st.file_uploader("Choose a text file", type=["txt"])
    if uploaded_file is not None:
        # Read and display the uploaded file
        text = uploaded_file.read().decode("utf-8")
        st.text_area("Uploaded Text", value=text, height=300)
        input_text = st.text_input("Enter a sequence of words from the uploaded text")
        if st.button("Predict Next Word from Uploaded Text"):
            max_sequence_len = model.input_shape[1] + 1
            next_word = predict_next_word(model, tokenizer, input_text, max_sequence_len)
            st.write(f'Next word: **{next_word}**')

elif choice == "About":
    st.subheader("About This App")
    st.markdown(
        """
        This app uses an LSTM model to predict the next word in a sequence.  
        You can either type a sequence of words or upload a text file for prediction.  
        Built with ❤️ using [Streamlit](https://streamlit.io/).
        """
    )

# Footer
st.markdown("---")
st.markdown("© 2025 Next Word Prediction App | Built with Streamlit")