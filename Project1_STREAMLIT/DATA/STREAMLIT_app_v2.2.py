import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sentence_transformers import SentenceTransformer, util
import plotly.express as px
import plotly.graph_objects as go

# Set page configuration
st.set_page_config(page_title="Customer Feedback Analysis", page_icon=":bar_chart:")

# Initialize the VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Initialize Sentence-BERT model for embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

# Define initial category descriptions
categories = {
    'Delivery Service': 'Comments related to the process of delivering products to customers. This includes feedback on shipping speed, tracking information, delivery personnel, and overall satisfaction with the delivery process. Common themes include timely delivery, packaging condition, and communication regarding delivery status. Keywords: delivery, shipping, arrived, package, courier, on-time, tracking, late, damaged, packaging, delay.',
    'Product Quality': 'Comments focusing on the quality of the products received. This includes feedback on the durability, materials used, craftsmanship, design, and overall performance of the product. Customers often mention their satisfaction or dissatisfaction with how the product meets their expectations. Keywords: quality, durable, material, design, built, performance, defect, broken, high-quality, craftsmanship, satisfaction.',
    'Customer Support': 'Comments pertaining to the interaction with customer support services. This includes feedback on the responsiveness, helpfulness, and overall effectiveness of the support team. Common themes include the resolution of issues, communication clarity, and the professionalism of support representatives. Keywords: support, service, help, assistance, response, resolution, professional, helpful, slow, unresponsive, courteous.'
}

# Generate embeddings for the category descriptions
category_embeddings = {category: model.encode(description) for category, description in categories.items()}

# Function to analyze sentiment
def analyze_sentiment(text):
    score = analyzer.polarity_scores(text)
    return score['compound']

# Function to get sentiment label
def get_sentiment_label(score):
    if score >= 0.05:
        return 'Positive'
    elif score <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

# Function to categorize review using embeddings
def categorize_review(text_embedding, category_embeddings, threshold=0.5):
    categories = []
    for category, category_embedding in category_embeddings.items():
        similarity = util.pytorch_cos_sim(text_embedding, category_embedding).item()
        if similarity >= threshold:
            categories.append(category)
    if not categories:
        categories.append('Other')
    return categories

# Streamlit app
st.title('Customer Feedback Analysis')

# Sidebar for file upload and date filtering
st.sidebar.header("Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=['csv'])

if uploaded_file is not None:
    # Specify the date format
    feedback_df = pd.read_csv(uploaded_file, parse_dates=['date'])
    feedback_df['date'] = pd.to_datetime(feedback_df['date'], errors='coerce')
    feedback_df = feedback_df.dropna(subset=['date'])
    feedback_df['week'] = feedback_df['date'].dt.isocalendar().week
    feedback_df['week_label'] = feedback_df['date'].dt.strftime('%V')

    # Calculate embeddings for all reviews once
    feedback_df['embedding'] = feedback_df['review_text'].apply(lambda x: model.encode(x))

    # Create the 'Category' column once using the pre-calculated embeddings
    feedback_df['Category'] = feedback_df['embedding'].apply(lambda x: categorize_review(x, category_embeddings))

    # Explode categories for multi-category reviews
    exploded_feedback_df = feedback_df.explode('Category')

    # Date filter
    min_date = feedback_df['date'].min()
    max_date = feedback_df['date'].max()
    start_date, end_date = st.sidebar.date_input(
        "Filter by date",
        [min_date, max_date],
        min_value=min_date,
        max_value=max_date
    )

    feedback_df = feedback_df[(feedback_df['date'] >= pd.to_datetime(start_date)) & (feedback_df['date'] <= pd.to_datetime(end_date))]
    exploded_feedback_df = exploded_feedback_df[(exploded_feedback_df['date'] >= pd.to_datetime(start_date)) & (exploded_feedback_df['date'] <= pd.to_datetime(end_date))]

    feedback_df['Sentiment Score'] = feedback_df['review_text'].apply(analyze_sentiment)
    feedback_df['Sentiment'] = feedback_df['Sentiment Score'].apply(get_sentiment_label)
    exploded_feedback_df['Sentiment Score'] = exploded_feedback_df['review_text'].apply(analyze_sentiment)
    exploded_feedback_df['Sentiment'] = exploded_feedback_df['Sentiment Score'].apply(get_sentiment_label)

    # Sentiment score filter
    st.sidebar.header("Filter by Sentiment")
    selected_sentiment = st.sidebar.multiselect('Select Sentiment', feedback_df['Sentiment'].unique(), feedback_df['Sentiment'].unique())
    feedback_df = feedback_df[feedback_df['Sentiment'].isin(selected_sentiment)]
    exploded_feedback_df = exploded_feedback_df[exploded_feedback_df['Sentiment'].isin(selected_sentiment)]

    custom_filter = st.sidebar.checkbox("Custom sentiment filtering")
    if custom_filter:
        # Sentiment score range filter
        min_score, max_score = st.sidebar.slider('Sentiment Score Range', -1.0, 1.0, (-1.0, 1.0))
        feedback_df = feedback_df[(feedback_df['Sentiment Score'] >= min_score) & (feedback_df['Sentiment Score'] <= max_score)]
        exploded_feedback_df = exploded_feedback_df[(exploded_feedback_df['Sentiment Score'] >= min_score) & (exploded_feedback_df['Sentiment Score'] <= max_score)]

        # Exclusion filter
        exclude_min, exclude_max = st.sidebar.slider('Exclude Sentiment Score Range', -1.0, 1.0, (-0.2, 0.2))
        feedback_df = feedback_df[~((feedback_df['Sentiment Score'] >= exclude_min) & (feedback_df['Sentiment Score'] <= exclude_max))]
        exploded_feedback_df = exploded_feedback_df[~((exploded_feedback_df['Sentiment Score'] >= exclude_min) & (exploded_feedback_df['Sentiment Score'] <= exclude_max))]

    # Category filter
    st.sidebar.header("Filter by Category")
    selected_category = st.sidebar.multiselect('Select Category', exploded_feedback_df['Category'].unique(), exploded_feedback_df['Category'].unique())
    exploded_feedback_df = exploded_feedback_df[exploded_feedback_df['Category'].isin(selected_category)]

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(["Feedback Overview", "Sentiment Analysis", "View & Download Data", "Manual Feedback Check"])

# Feedback Overview Tab
with tab1:
    if uploaded_file is not None:
        st.subheader('Volume of Feedback Over Time')
        volume_trends = feedback_df.groupby('week_label').size().reset_index(name='Number of Comments')

        fig = px.line(volume_trends, x='week_label', y='Number of Comments')

        # Filter week labels to show only multiples of 5
        week_labels = volume_trends['week_label']
        tickvals = [week for week in week_labels if int(week) % 5 == 0]

        # Update layout for better readability and to show week labels in 5-week steps
        fig.update_layout(
            xaxis_title='Calendar Week',
            yaxis_title='Number of Comments',
            xaxis=dict(
                tickmode='array',
                tickvals=tickvals,  # Show every 5th week label
                ticktext=[str(week) for week in tickvals]
            )
        )

        st.plotly_chart(fig)

        st.subheader('Total Comments by Category')
        show_relative = st.checkbox('Show Relative (%)', value=False)
        total_comments_by_category = exploded_feedback_df.groupby(['week_label', 'Category']).size().unstack().fillna(0)

        if show_relative:
            total_comments_by_week = exploded_feedback_df.groupby('week_label').size()
            total_comments_by_category = total_comments_by_category.div(total_comments_by_week, axis=0).multiply(100)

        fig = go.Figure()

        for category in total_comments_by_category.columns:
            fig.add_trace(go.Bar(
                x=total_comments_by_category.index,
                y=total_comments_by_category[category],
                name=category
            ))

        # Set tick values to show every fifth week
        tickvals = [week for week in total_comments_by_category.index if int(week) % 5 == 0]

        fig.update_layout(
            barmode='stack',
            xaxis_title='Calendar Week',
            yaxis_title='Total Comments' if not show_relative else 'Percentage (%)',
            xaxis_tickangle=-45,
            xaxis=dict(
                tickmode='array',
                tickvals=tickvals
            )
        )

        st.plotly_chart(fig)

        st.write("*Note: Values in this graph can be counted multiple times as reviews can be assigned to several categories.")
    else:
        st.warning("Please upload data for analysis")

# Sentiment Analysis Tab
with tab2:
    st.header('Sentiment Analysis')
    
    if uploaded_file is not None:
        # Average Sentiment Score
        
        avg_sentiment_score = feedback_df['Sentiment Score'].mean()
        st.metric("Average Sentiment Score", f"{avg_sentiment_score:.2f}")

        st.subheader('Sentiment Trends Over Time')
        sentiment_trends = feedback_df.groupby('week_label')['Sentiment Score'].mean().reset_index()

        fig = px.line(sentiment_trends, x='week_label', y='Sentiment Score')

        # Filter week labels to show only multiples of 5
        week_labels = sentiment_trends['week_label']
        tickvals = [week for week in week_labels if int(week) % 5 == 0]

        # Update layout for better readability and to show week labels in 5-week steps
        fig.update_layout(
            xaxis_title='Calendar Week',
            yaxis_title='Average Sentiment Score',
            xaxis_tickangle=-45,
            xaxis=dict(
                tickmode='array',
                tickvals=tickvals,
                ticktext=[str(week) for week in tickvals]
            )
        )

        st.plotly_chart(fig)

        st.subheader('Sentiment Distribution')
        sentiment_counts = feedback_df['Sentiment'].value_counts()
        total_counts = feedback_df['Sentiment'].count()
        sentiment_labels = [f'{count} ({(count / total_counts * 100):.1f}%)' for count in sentiment_counts]

        # Create a bar chart for sentiment distribution using Plotly
        fig = go.Figure(data=[
            go.Bar(
                x=sentiment_counts.index,
                y=sentiment_counts,
                text=sentiment_labels,
                textposition='auto',
                marker_color=['green', 'red', 'gray']
            )
        ])
        
        fig.update_layout(
           
            xaxis_title='Sentiment',
            yaxis_title='Count'
        )

        st.plotly_chart(fig)
    else:
        st.warning("Please upload data for analysis")

# Manual Feedback Check Tab
with tab4:
    st.header('Manual Feedback Check')

    if uploaded_file is not None:
        input_text = st.text_area('Enter individual feedback text')
        if input_text:
            text_embedding = model.encode(input_text)
            score = analyze_sentiment(input_text)
            label = get_sentiment_label(score)
            categories = categorize_review(text_embedding, category_embeddings)
            st.write(f'Sentiment Score: {score:.2f}')
            st.write(f'Sentiment: {label}')
            st.write(f'Categories: {", ".join(categories)}')
    else:
        st.warning("Please upload data for analysis")

# View & Download Data Tab
with tab3:
    st.header('View & Download Data')

    if uploaded_file is not None:
        st.subheader('Total Filtered Feedback Data')
        st.write(exploded_feedback_df)
        # Download filtered data as CSV
        csv = exploded_feedback_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Filtered Data as CSV",
            data=csv,
            file_name='filtered_feedback_data.csv',
            mime='text/csv',
        )
        
        st.subheader('Best 10 Filtered Feedbacks')
        most_positive_comments = feedback_df.nlargest(10, 'Sentiment Score')[['date', 'review_text', 'Sentiment Score']]
        st.write(most_positive_comments)
        most_positive_csv = most_positive_comments.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download 10 Most Positive Comments",
            data=most_positive_csv,
            file_name='most_positive_comments.csv',
            mime='text/csv',
        )

        st.subheader('Worst 10 Filtered Feedbacks')
        most_negative_comments = feedback_df.nsmallest(10, 'Sentiment Score')[['date', 'review_text', 'Sentiment Score']]
        st.write(most_negative_comments)
        most_negative_csv = most_negative_comments.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download 10 Most Negative Comments",
            data=most_negative_csv,
            file_name='most_negative_comments.csv',
            mime='text/csv',
        )

    else:
        st.warning("Please upload data for analysis")
