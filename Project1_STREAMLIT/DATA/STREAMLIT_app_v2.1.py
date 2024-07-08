#Application code -------------------------------------------------------------------------------------------------------------------------------------------


import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# Set page configuration
st.set_page_config(page_title="Customer Feedback Analysis", page_icon=":bar_chart:")

# Initialize the VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Define initial similar words for each category
initial_category_keywords = {
    'Delivery Service': ['delivery', 'shipping', 'arrived', 'package', 'courier'],
    'Product Quality': ['quality', 'durable', 'material', 'design', 'built'],
    'Customer Support': ['support', 'service', 'help', 'assistance', 'response']
}

# Create a dictionary to store category keywords dynamically
category_keywords = {category: set(keywords) for category, keywords in initial_category_keywords.items()}

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

# Function to categorize review
def categorize_review(text, keywords_dict):
    categories = []
    text_lower = text.lower()
    for category, keywords in keywords_dict.items():
        if any(keyword in text_lower for keyword in keywords):
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

    # Create the 'Category' column
    feedback_df['Category'] = feedback_df['review_text'].apply(lambda x: categorize_review(x, category_keywords))
    
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

    # Custom category keywords input
    custom_category = st.sidebar.checkbox("Custom category")
    if custom_category:
        st.sidebar.header("Category Keywords")
        for category in category_keywords.keys():
            st.sidebar.write(f"Keywords for {category}")
            selected_keywords = st.sidebar.multiselect(
                f"Select keywords for {category}", 
                initial_category_keywords[category], 
                list(category_keywords[category])
            )
            new_keyword = st.sidebar.text_input(f"Add new keyword for {category}", key=f"new_keyword_{category}")
            if new_keyword:
                category_keywords[category].add(new_keyword)
                st.sidebar.success(f"Added keyword: {new_keyword}")
                st.experimental_rerun()
            category_keywords[category] = set(selected_keywords)
        feedback_df['Category'] = feedback_df['review_text'].apply(lambda x: categorize_review(x, category_keywords))
        exploded_feedback_df = feedback_df.explode('Category')
       
        exploded_feedback_df = exploded_feedback_df[exploded_feedback_df['Category'].isin(selected_category)]

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(["Feedback Overview", "Sentiment Analysis", "View & Download Data", "Manual Feedback Check"])

# Feedback Overview Tab
with tab1:
    

    if uploaded_file is not None:
        
        import plotly.express as px
        import streamlit as st
        import pandas as pd

        # Assuming feedback_df is already defined and preprocessed

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
        # Assuming exploded_feedback_df is already defined and preprocessed

       
        import plotly.graph_objects as go
        import streamlit as st
        import pandas as pd

        # Assuming exploded_feedback_df is already defined and preprocessed

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
       
        import plotly.express as px
        import streamlit as st
        import pandas as pd

        # Assuming feedback_df is already defined and preprocessed

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
            score = analyze_sentiment(input_text)
            label = get_sentiment_label(score)
            categories = categorize_review(input_text, category_keywords)
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
