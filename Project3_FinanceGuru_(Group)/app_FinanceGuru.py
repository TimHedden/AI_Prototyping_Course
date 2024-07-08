import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import fitz  # pip install PyMuPDF
import openai  # OpenAI library
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI, OpenAI
from langchain.tools import DuckDuckGoSearchResults
from langchain.memory import ConversationBufferMemory
from io import StringIO, BytesIO
import io
from typing import List
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_core.prompts import MessagesPlaceholder
from langchain_experimental.agents.agent_toolkits.pandas.base import _get_functions_single_prompt
from langchain_experimental.tools.python.tool import PythonAstREPLTool
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables.history import RunnableWithMessageHistory

### Summary and Step-by-Step Description

#### Imports and Initial Setup
#- The script begins by importing necessary libraries such as Streamlit for the web interface, Pandas for data manipulation, Plotly for visualizations, PyMuPDF for PDF processing, and OpenAI for interacting with the OpenAI API.
#- The OpenAI API key is initialized for further use in the script.

#### Function Definitions
#1. **`get_category_colors(categories)`**:
#   - Generates a color palette for different transaction categories to be used in the Sankey chart.

#2. **`extract_text_from_pdf_with_pymupdf(pdf_file)`**:
#   - Extracts text from an uploaded PDF file using the PyMuPDF library.

#3. **`convert_text_to_csv(text, api_key)`**:
#   - Converts extracted text from the PDF into a CSV format using OpenAI's GPT model. This function utilizes a large language model (LLM) to parse and structure the text data into predefined categories.

#4. **`save_csv_to_dataframe(csv_content)`**:
#   - Converts the CSV content into a Pandas DataFrame and ensures the correct date format.

#### Initial Data Load and Session State Initialization
#- Loads initial data from a CSV file and sets it up in the session state.
#- Initializes session state variables for account balances and merged data.

#### Sidebar Setup for File Uploads and Filters
#- Provides options in the sidebar to upload CSV or Excel files and set account names.
#- Allows users to upload PDF files and convert them into transaction data.
#- Filters data based on selected years and months from the sidebar.

#### Main App Setup
#1. **Overview Tab**:
#   - Displays the total account balance and the percentage change from the previous month.
#   - Shows a line chart of account balances over the last six months using Plotly.
#   - Lists all account balances in a table.

#2. **Category Breakdown Tab**:
#   - Shows detailed breakdowns of income and expenses by category.
#   - Provides downloadable Excel files for transactions in each category.
#   - Displays progress towards monthly budget limits.

#3. **Analytics Tab**:
#   - Contains visualizations such as a grouped bar chart for monthly income and expenses, a Sankey diagram for income vs. expenses, and a donut chart for expenditure by category.

#4. **Chatbot Tab**:
#   - Implements a chatbot that can answer questions about spending habits using LangChain and OpenAI.
#   - Utilizes LLMs to process user queries and provide responses based on the transaction data.
#   - Initializes the chat history and provides an interface for user interactions.
#   - Includes a reset button to clear the chat history.

#5. **Transactions Tab**:
#   - Displays the filtered transaction data in a table.
#   - Provides a downloadable Excel file for the filtered transactions.

### Key Features
#- **Automatic Categorization**: Uses OpenAI's GPT model to automatically categorize transaction types based on descriptions.
#- **Large Language Models (LLMs)**: Employs LLMs to parse text from PDFs and convert it into structured data, and to power the chatbot for interactive financial insights.
#- **Comprehensive Data Integration**: Supports CSV, Excel, and PDF files, integrating various data sources into a unified financial analysis tool.
#- **Interactive Visualizations**: Provides detailed and interactive visualizations for income, expenses, and budget tracking.
#- **User-Friendly Interface**: Offers a user-friendly interface for uploading data, viewing analytics, and interacting with the chatbot.

# Initialize OpenAI API
api_key = "sk-proj-9YCfUiBd2nTx17G1UfbpT3BlbkFJRY3XTnVZrwmq15A3wnVG"

# Defining functions for the application -----------------------------------------------------------------------------------------------------------

# Add this function to get the color for each category --> Sankey Chart
def get_category_colors(categories):
    # Define the base color palette with red, yellow, orange, and purple shades
    base_palette = ['rgba(28, 150, 210, 0.8)', '#EF9A9A', '#FFD700', '#FFA500',
                    '#E57373', '#EF5350', '#9C27B0', '#F44336', '#E53935', '#D32F2F', 'grey']
    # Ensure the palette is large enough for the categories by repeating it
    extended_palette = base_palette * ((len(categories) + len(base_palette) - 1) // len(base_palette))
    # Insert green as the second color
    extended_palette.insert(1, 'green')
    # Slice the palette to match the number of categories
    final_palette = extended_palette[:len(categories)]
    color_map = {category: color for category, color in zip(categories, final_palette)}
    return [color_map.get(category, 'grey') for category in categories]


# Function to extract text from PDF using PyMuPDF
def extract_text_from_pdf_with_pymupdf(pdf_file):
    text = ""
    pdf_file.seek(0)  # Ensure the file pointer is at the start
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    for page in doc:
        text += page.get_text()
    return text


# Function to convert text to CSV using OpenAI
def convert_text_to_csv(text, api_key):
    client = openai.Client(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": '''Please parse the following text into CSV format with four columns: date, information, debit, credit and category. 
                Classify the transaction type based on this information into one of the following categories: Groceries, Dining, Travel, Shopping, 
                Utilities, Transportation, Entertainment, Services, Insurance, Other. Use a comma as the separator. Make it compatible to edit with 
                python language. Every line/row always ends with a comma no matter what. Ensure that: Information: Capture the essential content part 
                of the transaction description including where and what. Debit: Record amounts with a dot as the decimal separator. Amount over thousand 
                make sure it is 1039.50 instead of 1.039,50 Credit: Record amounts with a dot as the decimal separator. If a field has no value, leave it 
                empty as ',,' and ensure each row has exactly 5 values and separated by comma. 
                Example format with date, information, debit, credit: '23.05.2024,Gutsch.Eilüberw.n.PVO XU ZHENGRONG,,5975.00,Other,' where there is empty value for debit account '08.05.2024,Gutschrift Überweisung Alex Cappellari,,3.50,' instead of '08.05.2024,Gutschrift Überweisung Alex Cappellari,,3.50' in one row '21.05.2024,entgeltfreie Buchung PLAYTOMIC.IO,9.49,,' instead of '21.05.2024,entgeltfreie Buchung PLAYTOMIC.IO,9.49,' when there is empty value for credit. 
                Neglect any other information that does not involve transactions.'''
            },
            {
                "role": "user",
                "content": text
            }
        ],
        temperature=0,
        max_tokens=4095,
        top_p=1
    )
    csv_content = response.choices[0].message.content[7:-4]
    return csv_content


# Function to save CSV content to DataFrame
def save_csv_to_dataframe(csv_content):
    data = StringIO(csv_content)
    df = pd.read_csv(data, sep=",", header=None)
    # drop the last column
    df = df.iloc[:, :5]
    df.columns = df.iloc[0]  # set the first row as column names
    df = df[1:].reset_index(drop=True)  # drop the first row and reset the index
    # Correct the date format (assume day first)
    df['date'] = pd.to_datetime(df['date'], format='%d.%m.%Y', errors='coerce')
    return df


# Set up for total app -----------------------------------------------------------------------------------------------------------


# Load initial data
data_path = "RealBankData1.csv"
initial_data = pd.read_csv(data_path)
initial_data['Account'] = 'Bank Sabadell'

# Assuming your dates are in day/month/year format
initial_data['date'] = pd.to_datetime(initial_data['date'], dayfirst=True, errors='coerce')

# Initialize session state for account balances
if 'account_balances' not in st.session_state:
    st.session_state['account_balances'] = {'Bank Sabadell': initial_data['credit'].sum() - initial_data['debit'].sum()}
if 'merged_df' not in st.session_state:
    st.session_state['merged_df'] = initial_data


# Sidebar setup -----------------------------------------------------------------------------------------------------------

st.sidebar.title('Add Accounts')

# Sidebar for uploading files
uploaded_file = st.sidebar.file_uploader("Upload your CSV or Excel file", type=['csv', 'xlsx'])
account_name = st.sidebar.text_input("Set the account name")

if st.sidebar.button("Add transactions"):
    if uploaded_file and account_name:
        if uploaded_file.name.endswith('.csv'):
            new_data = pd.read_csv(uploaded_file)
        else:
            new_data = pd.read_excel(uploaded_file)

        # Ensure the date column is in datetime format
        new_data['date'] = pd.to_datetime(new_data['date'], dayfirst=True, errors='coerce')

        if {'debit', 'credit', 'date'}.issubset(new_data.columns):
            new_data['Account'] = account_name
            st.session_state['merged_df'] = pd.concat([st.session_state['merged_df'], new_data], ignore_index=True)
            st.session_state['merged_df'].dropna(subset=['date'], inplace=True)

            if account_name not in st.session_state['account_balances']:
                st.session_state['account_balances'][account_name] = 0

            account_balance = new_data['credit'].sum() - new_data['debit'].sum()
            st.session_state['account_balances'][account_name] += account_balance

            st.sidebar.success(f"Account {account_name} added with balance {account_balance:.2f}")
        else:
            st.sidebar.error("Uploaded file must contain 'debit', 'credit', and 'date' columns")
    elif not uploaded_file:
        st.sidebar.error("Please upload a file")
    elif not account_name:
        st.sidebar.error("Please enter an account name")

# Separate section for PDF upload and processing
uploaded_pdf = st.sidebar.file_uploader("Upload your PDF file", type=['pdf'])
pdf_account_name = st.sidebar.text_input("Set the account name (PDF)")

if st.sidebar.button("Add transactions (PDF)"):
    if uploaded_pdf and pdf_account_name:
        pdf_text = extract_text_from_pdf_with_pymupdf(uploaded_pdf)
        csv_content = convert_text_to_csv(pdf_text, api_key)
        new_data = save_csv_to_dataframe(csv_content)

        # Ensure the date column is in datetime format
        new_data['date'] = pd.to_datetime(new_data['date'], errors='coerce')
        new_data['credit'] = pd.to_numeric(new_data['credit'], errors='coerce')
        new_data['debit'] = pd.to_numeric(new_data['debit'], errors='coerce')
        new_data['location'] = ''  # Add location column with empty values

        if {'debit', 'credit', 'date'}.issubset(new_data.columns):
            new_data['Account'] = pdf_account_name
            st.session_state['merged_df'] = pd.concat([st.session_state['merged_df'], new_data], ignore_index=True)
            st.session_state['merged_df'].dropna(subset=['date'], inplace=True)

            if pdf_account_name not in st.session_state['account_balances']:
                st.session_state['account_balances'][pdf_account_name] = 0

            account_balance = new_data['credit'].sum() - new_data['debit'].sum()
            st.session_state['account_balances'][pdf_account_name] += account_balance

            st.sidebar.success(f"PDF Account {pdf_account_name} added with balance {account_balance:.2f}")
        else:
            st.sidebar.error("Uploaded PDF must contain 'debit', 'credit', and 'date' columns")
    elif not uploaded_pdf:
        st.sidebar.error("Please upload a PDF file")
    elif not pdf_account_name:
        st.sidebar.error("Please enter an account name for the PDF")

# Get unique years and months from the data
unique_years = st.session_state['merged_df']['date'].dt.year.dropna().unique()
unique_months = st.session_state['merged_df']['date'].dt.month.dropna().unique()
month_to_name = {1: 'January', 2: 'February', 3: 'March', 4: 'April', 5: 'May', 6: 'June', 7: 'July', 8: 'August',
                 9: 'September', 10: 'October', 11: 'November', 12: 'December'}
unique_month_names = ['Show all months'] + [month_to_name[month] for month in sorted(unique_months)]

# Sidebar filters
st.sidebar.title('Filters')
st.sidebar.markdown('(Does not apply to Overview and Transaction tab)')
selected_years = st.sidebar.multiselect('Select years:', sorted(unique_years), default=sorted(unique_years))
selected_months = st.sidebar.multiselect('Select months:', unique_month_names, default=unique_month_names)

if 'Show all months' in selected_months:
    selected_months = list(month_to_name.values())

month_to_num = {name: num for num, name in month_to_name.items()}
selected_month_nums = [month_to_num[month] for month in selected_months]

# Filter data based on sidebar selections
filtered_data = st.session_state['merged_df'][
    (st.session_state['merged_df']['date'].dt.year.isin(selected_years)) &
    (st.session_state['merged_df']['date'].dt.month.isin(selected_month_nums))
    ]



# Main app setup -----------------------------------------------------------------------------------------------------------


# Main page navigation

st.title('FinanceGuru')
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ['Overview', 'Category Breakdown', 'Analytics', 'Chatbot', 'Transactions'])

# Overview Tab
with tab1:
    # Calculate the balance based on actual data
    total_credit = st.session_state['merged_df']['credit'].sum() if 'credit' in st.session_state['merged_df'] else 0
    total_debit = st.session_state['merged_df']['debit'].sum() if 'debit' in st.session_state['merged_df'] else 0
    total_balance = total_credit - total_debit

    st.header(f'Your balance is {total_balance:.2f} EUR')

    # Calculate the monthly change based on actual data
    st.session_state['merged_df']['date'] = pd.to_datetime(st.session_state['merged_df']['date'], errors='coerce')
    st.session_state['merged_df'].dropna(subset=['date'], inplace=True)

    today = pd.Timestamp.today().date()
    if today in st.session_state['merged_df']['date'].dt.date.values:
        current_date = pd.Timestamp(today)
    else: # If no data in dataset, assuming the current date is 2024-06-30 by default (just for demo purpose)
        current_date = pd.Timestamp('2024-06-30')

    last_month_date = current_date - pd.DateOffset(months=1)

    # Filter data for the current and previous month
    current_month = st.session_state['merged_df'][
        st.session_state['merged_df']['date'].dt.to_period('M') == current_date.to_period('M')
        ] if not st.session_state['merged_df'].empty else pd.DataFrame()
    previous_month = st.session_state['merged_df'][
        st.session_state['merged_df']['date'].dt.to_period('M') == last_month_date.to_period('M')
        ] if not st.session_state['merged_df'].empty else pd.DataFrame()

    current_month_balance = current_month['credit'].sum() - current_month[
        'debit'].sum() if not current_month.empty else 0
    previous_month_balance = previous_month['credit'].sum() - previous_month[
        'debit'].sum() if not previous_month.empty else 0
    monthly_change = (abs(current_month_balance) - abs(previous_month_balance)) if abs(previous_month_balance) != 0 else 0

    font_color = 'green' if monthly_change > 0 else 'red'
    font_size = '20px'
    balance_change_text = 'Your balance has changed by '
    percentage_text = f'<span style="color: {font_color};">{monthly_change:.2f} EUR</span> since last month (exact-date MoM)'
    st.markdown(f'<p style="font-size: {font_size};">{balance_change_text}{percentage_text}</p>',
                unsafe_allow_html=True)

    # Assuming current_date is defined somewhere in your code
    current_date = pd.Timestamp.now()  # or use the current date variable you have

    # Plotly line chart showing account balance over the last 6 months
    # Filter data for the last 6 months
    last_six_months_start_dates = pd.date_range(end=current_date, periods=6, freq='MS')  # Start of the month

    # Initialize an empty DataFrame to hold monthly balances
    monthly_balances = pd.DataFrame(columns=['date', 'balance'])

    # Calculate the balance for each start of the month in the last six months
    for date in last_six_months_start_dates:
        monthly_data = st.session_state['merged_df'][st.session_state['merged_df']['date'] < date]
        balance = monthly_data['credit'].sum() - monthly_data['debit'].sum()
        monthly_balances = monthly_balances.append({'date': date, 'balance': balance}, ignore_index=True)

    # Calculate the balance up to the current date
    current_data = st.session_state['merged_df'][st.session_state['merged_df']['date'] <= current_date]
    current_balance = current_data['credit'].sum() - current_data['debit'].sum()
    current_date_balance = pd.DataFrame({
        'date': [current_date],
        'balance': [current_balance]
    })

    # Combine the monthly balances with the current date balance
    combined_balance = pd.concat([monthly_balances, current_date_balance], ignore_index=True)

    st.markdown('Monthly trend:')


    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=combined_balance['date'], y=combined_balance['balance'], mode='lines+markers', name='Balance')
    )
    fig.update_layout(
        xaxis=dict(
            title='Month',
            tickformat='%Y-%m-%d',
            tickmode='array',
            tickvals=[date for date in last_six_months_start_dates] + [current_date],
            ticktext=[date.strftime('%Y-%m-%d') for date in last_six_months_start_dates] + [current_date.strftime('%Y-%m-%d')]
        ),
        yaxis=dict(title='Balance (EUR)')
    )
    st.plotly_chart(fig)

    # Calculate the total budget set in the categories
    total_budget = 0
    categories = ['Transportation', 'Services', 'Dining', 'Entertainment', 'Education', 'Utilities', 'Shopping',
                  'Groceries', 'Insurance', 'Other']
    for i, category in enumerate(categories):
        if f'budget_{i}' in st.session_state:
            total_budget += st.session_state[f'budget_{i}']

    # Get the current month name
    current_month_name = current_date.strftime('%B')

    st.header("Your Accounts")
    accounts_df = pd.DataFrame(list(st.session_state['account_balances'].items()), columns=['Account', 'Balance'])
    accounts_df['Balance'] = accounts_df['Balance'].apply(lambda x: f"{x:.2f}")
    st.table(accounts_df)

# Category Breakdown Tab
with tab2:
    tabs = st.tabs(
        ['💰Income', '💸Total Expenses', '🚗Transportation', '💈Services', '🍴Dining', '🎮Entertainment', '📖Education',
         '⚡Utilities',
         '🛒Shopping', '🍞Groceries', '💶Insurance', 'Other'])
    categories = ['Income', 'Total Expenses', 'Transportation', 'Services', 'Dining', 'Entertainment', 'Education',
                  'Utilities',
                  'Shopping', 'Groceries', 'Insurance', 'Other']

    for i, tab in enumerate(tabs):
        with tab:
            if categories[i] == 'Income':
                # Calculate average monthly income
                total_income = filtered_data['credit'].sum()
                num_months = len(filtered_data['date'].dt.to_period('M').unique())
                avg_monthly_income = total_income / num_months if num_months > 0 else 0
                st.subheader(f'Average Monthly Income: {avg_monthly_income:.2f} EUR')
                st.subheader(f'Total Income: {total_income:.2f} EUR')
                income_df = filtered_data[filtered_data['credit'] > 0][['date', 'information', 'location', 'credit']]
                income_df['credit'] = income_df['credit'].apply(lambda x: f"{x:.2f}")
                st.dataframe(income_df, height=600, width=1200)  # Scrollable transaction field

                # Add download button
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                    income_df.to_excel(writer, index=False)
                st.download_button(
                    label="Download transactions as Excel",
                    data=buffer,
                    file_name='income_transactions.xlsx',
                    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                )

            elif categories[i] == 'Total Expenses':
                # Calculate total expenses and average monthly expenses
                total_expense = filtered_data['debit'].sum()
                num_months = len(filtered_data['date'].dt.to_period('M').unique())
                avg_monthly_expense = total_expense / num_months if num_months > 0 else 0
                st.subheader(f'Average Monthly Expenses: {avg_monthly_expense:.2f} EUR')
                st.subheader(f'Total Expenses: {total_expense:.2f} EUR')
                expenses_df = filtered_data[filtered_data['debit'] > 0][['date', 'information', 'location', 'debit']]
                expenses_df['debit'] = expenses_df['debit'].apply(lambda x: f"{x:.2f}")
                st.dataframe(expenses_df, height=600, width=1200)  # Scrollable transaction field

                # Add download button
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                    expenses_df.to_excel(writer, index=False)
                st.download_button(
                    label="Download transactions as Excel",
                    data=buffer,
                    file_name='total_expenses_transactions.xlsx',
                    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                )

            else:
                category_df = filtered_data[filtered_data['category'] == categories[i]]
                category_debits = category_df[['date', 'information', 'location', 'debit']].copy()
                category_debits['debit'] = category_debits['debit'].apply(lambda x: f"{x:.2f}")

                # Calculate average monthly expense for the category
                total_expense = category_debits['debit'].astype(float).sum()
                num_months = len(filtered_data['date'].dt.to_period('M').unique())
                avg_monthly_expense = total_expense / num_months if num_months > 0 else 0
                st.subheader(f'Average Monthly {categories[i]} Expenses: {avg_monthly_expense:.2f} EUR')

                # Calculate the current month's expenditure
                # Check if today's date is in the dataframe
                today = pd.Timestamp.today().date()
                if today in category_df['date'].dt.date.values:
                    current_date = pd.Timestamp(today)
                else: # If no data in dataset, assuming the current date is 2024-06-30 by default
                    current_date = pd.Timestamp('2024-06-30')
                current_month_expense = category_df[category_df['date'].dt.month == current_date.month]['debit'].astype(
                    float).sum()
                
                # Budget logic
                budget_limit = st.number_input(
                    f'Enter your budget limit for {categories[i]} in Euro (default is avg monthly expenses)',
                    value=float(avg_monthly_expense),
                    key=f'budget_{i}')
                
                # Display the progress line for current month's expenditure as a percentage of the average monthly expense
                percentage_used = (current_month_expense / budget_limit) * 100 if avg_monthly_expense != 0 else 0
                # Check if the percentage_used is negative value
                if percentage_used < 0 or percentage_used > 100:
                    st.warning('You have exceeded your budget limit. Please consider revising your budget.')
                elif percentage_used == 0:
                    st.warning(f'No current expense in the category of {categories[i]} for the current month.')    
                else:
                    st.write(
                        f'{current_month_expense:.2f} EUR ({percentage_used:.2f}%) used of current Month\'s {categories[i]} Expenditure Budget')
                    st.progress(percentage_used / 100)

                

                # Scrollable transaction field
                st.subheader(f'Accumulated Expenses: {total_expense:.2f} EUR')
                st.dataframe(category_debits, height=600, width=1200)  # Scrollable transaction field

                # Add download button
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                    category_debits.to_excel(writer, index=False)
                st.download_button(
                    label="Download transactions as Excel",
                    data=buffer,
                    file_name=f'{categories[i].lower()}_transactions.xlsx',
                    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                )

# Analytics Tab
with tab3:
    # Grouped Bar chart -------------------------------------
    st.subheader('Total Monthly Income and Expenditure')
    if not filtered_data.empty:
        # Aggregate expenditure and income by month
        monthly_data = filtered_data.groupby(filtered_data['date'].dt.to_period('M')).agg({
            'credit': 'sum',
            'debit': 'sum'
        }).reset_index()
        # Convert the period to the first day of each month
        monthly_data['date'] = monthly_data['date'].dt.to_timestamp('M')

        # Create a Plotly figure
        fig = go.Figure()

        # Add bar for Expenditure
        fig.add_trace(go.Bar(x=monthly_data['date'], y=monthly_data['debit'], name='Expenditure'))

        # Add bar for Income
        fig.add_trace(go.Bar(x=monthly_data['date'], y=monthly_data['credit'], name='Income'))

        # Update the layout for a grouped bar chart
        fig.update_layout(
            barmode='group',
            width=700,
            height=500,
            xaxis=dict(
                tickformat='%Y-%m-%d',
                dtick='M1',
                ticklabelmode="period"
            )
        )

        # Display the figure in Streamlit
        st.plotly_chart(fig)

        # Sankey diagram -------------------------------
        st.subheader('Income vs. Expenses - Sankey Diagram')

        # Define categories
        income_category = 'Income'
        expense_categories = ['Transportation', 'Services', 'Dining', 'Entertainment', 'Education', 'Utilities',
                              'Shopping', 'Groceries', 'Insurance', 'Other']

        # Calculate total income
        total_income = filtered_data['credit'].sum()

        # Calculate expenses for each category
        transportation_expense = abs(filtered_data[filtered_data['category'] == 'Transportation']['debit'].sum())
        services_expense = abs(filtered_data[filtered_data['category'] == 'Services']['debit'].sum())
        dining_expense = abs(filtered_data[filtered_data['category'] == 'Dining']['debit'].sum())
        entertainment_expense = abs(filtered_data[filtered_data['category'] == 'Entertainment']['debit'].sum())
        education_expense = abs(filtered_data[filtered_data['category'] == 'Education']['debit'].sum())
        utilities_expense = abs(filtered_data[filtered_data['category'] == 'Utilities']['debit'].sum())
        shopping_expense = abs(filtered_data[filtered_data['category'] == 'Shopping']['debit'].sum())
        groceries_expense = abs(filtered_data[filtered_data['category'] == 'Groceries']['debit'].sum())
        insurance_expense = abs(filtered_data[filtered_data['category'] == 'Insurance']['debit'].sum())
        other_expense = abs(filtered_data[filtered_data['category'] == 'Other']['debit'].sum())

        # Sum up all expenses
        total_expense = (transportation_expense + services_expense + dining_expense + entertainment_expense +
                         education_expense + utilities_expense + shopping_expense + groceries_expense +
                         insurance_expense + other_expense)

        # Calculate potential savings
        potential_savings = total_income - total_expense

        # Add potential savings header
        st.markdown(f'##### {potential_savings:.2f} EUR Potential Savings')

        # Define labels and values
        labels = [income_category, 'Potential Savings', 'Transportation', 'Services', 'Dining', 'Entertainment',
                  'Education', 'Utilities',
                  'Shopping', 'Groceries', 'Insurance', 'Other']
        values = [total_income, potential_savings, transportation_expense, services_expense, dining_expense,
                  entertainment_expense,
                  education_expense, utilities_expense, shopping_expense, groceries_expense, insurance_expense,
                  other_expense]
        colors = get_category_colors(labels)

        # Define nodes
        node = dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=labels,
            color=colors
        )

        # Define links
        source = [0] * (len(expense_categories) + 1)
        target = list(range(1, len(expense_categories) + 2))
        value = [potential_savings, transportation_expense, services_expense, dining_expense, entertainment_expense,
                 education_expense,
                 utilities_expense, shopping_expense, groceries_expense, insurance_expense, other_expense
                 ]

        link = dict(
            source=source,
            target=target,
            value=value,
            color=[colors[0] for _ in source]  # Use income category color for all links
        )

        fig_sankey = go.Figure(data=[go.Sankey(node=node, link=link)])
        st.plotly_chart(fig_sankey)

        # Donut Chart ---------------------------------
        st.subheader('Expenditure by Category')
        transportation_data = st.session_state['merged_df'][
            st.session_state['merged_df']['category'] == 'Transportation']
        services_data = st.session_state['merged_df'][st.session_state['merged_df']['category'] == 'Services']
        dining_data = st.session_state['merged_df'][st.session_state['merged_df']['category'] == 'Dining']
        entertainment_data = st.session_state['merged_df'][st.session_state['merged_df']['category'] == 'Entertainment']
        education_data = st.session_state['merged_df'][st.session_state['merged_df']['category'] == 'Education']
        utilities_data = st.session_state['merged_df'][st.session_state['merged_df']['category'] == 'Utilities']
        shopping_data = st.session_state['merged_df'][st.session_state['merged_df']['category'] == 'Shopping']
        groceries_data = st.session_state['merged_df'][st.session_state['merged_df']['category'] == 'Groceries']
        insurance_data = st.session_state['merged_df'][st.session_state['merged_df']['category'] == 'Insurance']
        other_data = st.session_state['merged_df'][st.session_state['merged_df']['category'] == 'Other']

        expenditures = {}

        for month in month_to_num.values():
            expenditures[month] = {
                'Transportation': abs(
                    transportation_data[transportation_data['date'].dt.month == month]['debit'].sum()),
                'Services': abs(services_data[services_data['date'].dt.month == month]['debit'].sum()),
                'Dining': abs(dining_data[dining_data['date'].dt.month == month]['debit'].sum()),
                'Entertainment': abs(entertainment_data[entertainment_data['date'].dt.month == month]['debit'].sum()),
                'Education': abs(education_data[education_data['date'].dt.month == month]['debit'].sum()),
                'Utilities': abs(utilities_data[utilities_data['date'].dt.month == month]['debit'].sum()),
                'Shopping': abs(shopping_data[shopping_data['date'].dt.month == month]['debit'].sum()),
                'Groceries': abs(groceries_data[groceries_data['date'].dt.month == month]['debit'].sum()),
                'Insurance': abs(insurance_data[insurance_data['date'].dt.month == month]['debit'].sum()),
                'Other': abs(other_data[other_data['date'].dt.month == month]['debit'].sum())
            }

        # If the user has selected at least one month on the sidebar, show the pie chart
        if selected_month_nums:
            aggregated_expenditures = {}
            for month in selected_month_nums:
                for category, amount in expenditures[month].items():
                    aggregated_expenditures[category] = aggregated_expenditures.get(category, 0) + amount
            # Sort the categories in pie chart from biggest to smallest weight
            sorted_categories = sorted(aggregated_expenditures.items(), key=lambda x: x[1], reverse=True)
            labels, values = zip(*sorted_categories)

            fig_pie = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.5)])
            fig_pie.update_traces(hovertemplate='%{label}: %{value:.1f} EUR<br>%{percent:.1%}')
            fig_pie.update_layout(

                width=800,  # Set the width of the figure
                height=600  # Set the height of the figure
            )
            st.plotly_chart(fig_pie)
        else:
            st.write("Please select at least one month.")
    else:
        st.write("No transactions uploaded yet.")


# Chatbot Tab
with tab4:
    df = st.session_state['merged_df']

    llm = OpenAI(api_key=api_key, temperature=0)

    prompt = _get_functions_single_prompt(df)
    prompt.input_variables.append("chat_history")
    prompt.messages.insert(1, MessagesPlaceholder(variable_name="chat_history"))

    tools = [PythonAstREPLTool(locals={"df": df})]

    chat_model = ChatOpenAI(api_key=api_key, model="gpt-3.5-turbo")
    agent = create_openai_functions_agent(chat_model, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)


    class InMemoryHistory(BaseChatMessageHistory, BaseModel):
        messages: List[BaseMessage] = Field(default_factory=list)

        def add_message(self, message: BaseMessage) -> None:
            self.messages.append(message)

        def clear(self) -> None:
            self.messages = []


    store = {}


    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in store:
            store[session_id] = InMemoryHistory()
        return store[session_id]


    chain = RunnableWithMessageHistory(  # RunnableWithMessageHistory
        agent_executor,
        get_session_history=get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )

    # Streamlit App
    
    st.subheader("Hello there!") 
    st.markdown("I am your personal finance chatbot. Feel free to ask me questions about your spending habits like: How much did I spend in Madrid in total? or How much more did I spent on groceries in April compared to May?")  # Set the title of the app

    # Initialize session state
    if "session_id" not in st.session_state:
        st.session_state["session_id"] = "default_session"
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    # Button to reset the session state
    if st.button("Reset session (until last message)"):
        # Clear the in-memory store for the session_id
        if st.session_state["session_id"] in store:
            store[st.session_state["session_id"]].clear()
        # Reset session state variables
        st.session_state["session_id"] = "default_session"
        st.session_state["chat_history"] = []

    # Input field for user messages
    user_input = st.text_input("New Message: ", key="user_input")

    # Process user input
    if user_input:
        st.session_state["chat_history"].append({"role": "user", "content": user_input})
        response = chain.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": st.session_state["session_id"]}},
        )
        st.session_state["chat_history"].append({"role": "assistant", "content": response["output"]})

    # Display chat history
    for msg in st.session_state["chat_history"]:  # Loop through each message in the chat history
        if msg["role"] == "user":
            st.text_area("You:", value=msg["content"], height=75, key=f"user_{msg['content']}")
        else:
            st.text_area("Assistant:", value=msg["content"], height=75, key=f"assistant_{msg['content']}")



# Transactions Tab
with tab5:
    # Calculate the number of transactions
    num_transactions = len(filtered_data)

    # Display the number of transactions before the headline
    st.subheader(f'{num_transactions} Transactions')

    # Display the transaction data
    st.dataframe(filtered_data)


    # Add a download button for the transactions as an Excel file
    @st.cache_data
    def convert_df_to_excel(df):
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='Transactions')
        return output.getvalue()


    excel_data = convert_df_to_excel(filtered_data)
    st.download_button(
        label="Download transactions as Excel",
        data=excel_data,
        file_name='transactions.xlsx',
        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )