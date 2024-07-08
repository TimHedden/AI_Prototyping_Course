
##Technical Description of Data Processing

#File Upload and Date Parsing:

#The application allows the user to upload a CSV file containing customer feedback.
#The date column in the CSV file is parsed and converted to a datetime format. Rows with invalid or missing dates are removed.

#Adding Calendar Week and Week Labels:

#A new column week is added to the DataFrame, which contains the calendar week number of each feedback date.
#Another column week_label is created, formatting the week number and year as a string (e.g., KW01 2024).

#Categorizing Feedback:

#Feedback text is categorized based on predefined keywords for different categories (e.g., Delivery Service, Product Quality).
#The feedback that matches multiple categories is exploded into multiple rows to handle multi-category reviews.

#Sentiment Analysis:

#The VADER sentiment analyzer is used to calculate a sentiment score for each feedback comment.
#Based on the sentiment score, each feedback is labeled as Positive, Negative, or Neutral.

#Filtering Data:

#Users can filter the feedback data by date range, sentiment, and category using the sidebar options.
#Additional custom filters allow for more granular sentiment score filtering, including excluding certain ranges of sentiment scores.

#Generating Graphs:

#Various graphs are generated to visualize the feedback data:
#Volume of Feedback Over Time: A line graph showing the number of feedback comments received per calendar week.
#Total Comments by Category: A stacked bar chart showing the number of comments per category for each calendar week. An option is provided to show this data as a percentage of total comments.
#Sentiment Trends Over Time: A line graph showing the average sentiment score per calendar week.
#Sentiment Distribution: A bar chart showing the count and percentage of Positive, Negative, and Neutral feedback.
#Manual Feedback Check and Data Download:

# A section for manually entering and analyzing individual feedback text.
#Options to view and download the filtered feedback data as a CSV file.



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
    st.header('Feedback Overview')

    if uploaded_file is not None:
        st.subheader('Volume of Feedback Over Time')
        volume_trends = feedback_df.groupby('week_label').size()

        fig, ax = plt.subplots()
        volume_trends.plot(kind='line', ax=ax, color='orange')
        ax.set_xlabel('Calendar Week')
        ax.set_ylabel('Number of Comments')
        ax.set_title('Volume of Feedback Over Time')
        ax.set_xticks(range(len(volume_trends.index)))
        ax.set_xticklabels(volume_trends.index, rotation=45, ha='right')
        st.pyplot(fig)

        st.subheader('Total Comments by Category')
        show_relative = st.checkbox('Show Relative (%)', value=False)
        total_comments_by_category = exploded_feedback_df.groupby(['week_label', 'Category']).size().unstack().fillna(0)

        if show_relative:
            total_comments_by_week = exploded_feedback_df.groupby('week_label').size()
            total_comments_by_category = total_comments_by_category.div(total_comments_by_week, axis=0).multiply(100)

        fig, ax = plt.subplots()
        total_comments_by_category.plot(kind='bar', stacked=True, ax=ax)
        ax.set_xlabel('Calendar Week')
        ax.set_ylabel('Total Comments' if not show_relative else 'Percentage (%)')
        ax.set_title('Total Comments by Category')
        ax.set_xticks(range(len(total_comments_by_category.index)))
        ax.set_xticklabels(total_comments_by_category.index, rotation=45, ha='right')
        st.pyplot(fig)

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
        sentiment_trends = feedback_df.groupby('week_label')['Sentiment Score'].mean()
        fig, ax = plt.subplots()
        sentiment_trends.plot(kind='line', ax=ax)
        ax.set_xlabel('Calendar Week')
        ax.set_ylabel('Average Sentiment Score')
        ax.set_title('Sentiment Trends Over Time')
        ax.set_xticks(range(len(sentiment_trends.index)))
        ax.set_xticklabels(sentiment_trends.index, rotation=45, ha='right')
        st.pyplot(fig)
        
        st.subheader('Sentiment Distribution')
        sentiment_counts = feedback_df['Sentiment'].value_counts()
        total_counts = feedback_df['Sentiment'].count()
        sentiment_labels = [f'{count} ({(count / total_counts * 100):.1f}%)' for count in sentiment_counts]
        fig, ax = plt.subplots()
        sentiment_counts.plot(kind='bar', ax=ax, color=['green', 'red', 'gray'])
        ax.set_xticklabels(sentiment_counts.index, rotation=0)
        for i, v in enumerate(sentiment_counts):
            ax.text(i, v + 3, sentiment_labels[i], ha='center')
        ax.set_xlabel('')
        ax.set_ylabel('Count')
        ax.set_title(f'Sentiment Distribution (Total: {total_counts})')
        st.pyplot(fig)
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
