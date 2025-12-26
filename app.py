import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import joblib
import os
import plotly.express as px 
import plotly.graph_objects as go 

# Load the model and vectorizer with caching to prevent reloading on every run
@st.cache_resource
def load_model():
    try:
        model = joblib.load(r'C:\Users\suyas\Downloads\sem_5_miniproject\sentiment_model.pkl')
        vectorizer = joblib.load(r'C:\Users\suyas\Downloads\sem_5_miniproject\tfidf_vectorizer.pkl')
        return model, vectorizer
    except Exception as e:
        st.sidebar.error(f"Error loading model or vectorizer: {e}")
        return None, None

model, vectorizer = load_model()

# Title of the Web App
st.title("Amazon Review Sentiment Analysis")

# Sidebar for user input
st.sidebar.header("User Input")

# User input review
user_review = st.sidebar.text_area("Enter a product review")

# Function to predict sentiment
def predict_sentiment(review):
    try:
        # Vectorizing the input review
        review_vector = vectorizer.transform([review])
        if review_vector.shape[1] != model.coef_.shape[1]:
            st.error(f"Model expects {model.coef_.shape[1]} features, but got {review_vector.shape[1]}. Please check the vectorizer.")
            return None
        prediction = model.predict(review_vector)
        return prediction[0]
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None

# Predict sentiment if a review is provided
if user_review:
    sentiment = predict_sentiment(user_review)
    if sentiment is not None:
        st.write(f"Predicted Sentiment: **{'Positive' if sentiment == 1 else 'Negative'}**")

# Option to upload dataset for batch analysis
uploaded_file = st.file_uploader("Upload a dataset for batch sentiment analysis", type=["csv"])

df = None

# Load the uploaded file or default file
if uploaded_file is not None:
    st.write(f"File name: {uploaded_file.name}")  # Log the uploaded file name
    st.write(f"File size: {uploaded_file.size} bytes")  # Log the file size
    try:
        df = pd.read_csv(uploaded_file)
        st.sidebar.success("File uploaded successfully!")
    except Exception as e:
        st.error(f"Error processing the uploaded file: {e}")
else:
    default_file_path = r"C:\Users\suyas\Downloads\sem_5_miniproject\Reviews.csv"
    if os.path.isfile(default_file_path):
        try:
            df = pd.read_csv(default_file_path)
            st.sidebar.success("Using default file.")
        except Exception as e:
            st.error(f"Error loading the default file: {e}")
    else:
        st.sidebar.error("Default file not found.")

# Ensure df is not None before proceeding
if df is not None:
    st.write(f"Loaded {df.shape[0]} rows.")  # Display the number of rows loaded
    st.write("Columns in uploaded file:", df.columns)  # Display column names

    # Ask the user to specify the column name for reviews
    review_column = st.sidebar.text_input("Enter the name of the column that contains reviews", "Text")

    if review_column in df.columns:
        st.write(f"Using column '{review_column}' for sentiment analysis.")

        # Limit the rows for testing purposes (e.g., first 100 rows)
        sample_df = df[[review_column]].head(100)
        
        # Perform batch sentiment analysis
        with st.spinner('Performing batch sentiment analysis...'):
            try:
                sample_df['Prediction'] = sample_df[review_column].apply(predict_sentiment)
                st.write(sample_df[[review_column, 'Prediction']])

                # Sentiment counts and percentages
                sentiment_counts = sample_df['Prediction'].value_counts(normalize=True)
                sentiment_counts_absolute = sample_df['Prediction'].value_counts()

                labels = ['Positive' if i == 1 else 'Negative' for i in sentiment_counts.index]

                # Interactive Bar Chart with Plotly
                fig_bar = px.bar(
                    x=labels,
                    y=sentiment_counts.values,
                    labels={'x': 'Sentiment', 'y': 'Percentage'},
                    title="Sentiment Distribution",
                )
                st.plotly_chart(fig_bar)

                # Interactive Pie Chart with Plotly
                fig_pie = go.Figure(data=[go.Pie(
                    labels=labels,
                    values=sentiment_counts_absolute.values,
                    hole=0.4,
                    hoverinfo="label+percent+value",
                    textinfo="label+percent",
                )])
                fig_pie.update_layout(title_text="Sentiment Percentage")
                st.plotly_chart(fig_pie)

                positive_texts = " ".join(sample_df[sample_df['Prediction'] == 1][review_column].values)
                negative_texts = " ".join(sample_df[sample_df['Prediction'] == 0][review_column].values)

                st.write("Word Cloud for Positive Reviews")
                positive_wordcloud = WordCloud(width=800, height=400, background_color='white').generate(positive_texts)
                plt.imshow(positive_wordcloud, interpolation='bilinear')
                plt.axis('off')
                st.pyplot()

                st.write("Word Cloud for Negative Reviews")
                negative_wordcloud = WordCloud(width=800, height=400, background_color='white').generate(negative_texts)
                plt.imshow(negative_wordcloud, interpolation='bilinear')
                plt.axis('off')
                st.pyplot()
            except Exception as e:
                st.error(f"Error during batch analysis: {e}")
    else:
        st.error(f"'{review_column}' column not found in the dataset.")
else:
    st.write("Please upload a dataset or check the default file for analysis.")

# Display the model accuracy
st.sidebar.header("Model Info")
st.sidebar.write("Model Accuracy: 0.94")
