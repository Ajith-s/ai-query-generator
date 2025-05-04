import streamlit as st
import pandas as pd
import duckdb
from openai import OpenAI
import os
from dotenv import load_dotenv
import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime
import matplotlib.pyplot as plt
import plotly.express as px
from uuid import uuid4


# Load environment variables
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Firebase initialization
if not firebase_admin._apps:
    cred = credentials.Certificate("firebase_service_account.json")  # Add your Firebase service account JSON file
    firebase_admin.initialize_app(cred)
db = firestore.client()

# Generate synthetic data for 5 tables

def create_dummy_data():
    users = pd.DataFrame({
        'user_id': range(1, 6),
        'name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
        'email': ['alice@example.com', 'bob@example.com', 'charlie@example.com', 'diana@example.com', 'eve@example.com'],
        'signup_date': pd.date_range('2024-01-01', periods=5, freq='M')
    })

    products = pd.DataFrame({
        'product_id': range(1, 6),
        'product_name': ['Shirt', 'Gadget', 'Book', 'Pen', 'Sweater'],
        'category': ['Clothing', 'Electronics', 'Books', 'Stationery', 'Clothing'],
        'description': ['A stylish shirt', 'A smart gadget', 'An interesting book', 'A useful pen', 'A warm sweater'],
        'price': [10.0, 20.0, 15.0, 25.0, 30.0]
    })

    orders = pd.DataFrame({
        'order_id': range(1, 17),
        'user_id': [1,2,3,4,5,2,3,4,5,1,6,4,5,1,2,3],
        'product_id': [1,2,3,4,5,2,3,4,5,1,6,4,5,1,2,3],
        'order_date': pd.date_range('2024-03-01', periods=16, freq='7D'),  # Fixed length to match other columns
        'status': ['completed', 'completed', 'pending', 'completed', 'failed', 'completed', 'pending', 'completed', 'failed', 'completed', 'completed', 'completed', 'pending', 'completed', 'failed', 'completed'],
        'quantity': [1,2,1,3,2,1,1,2,1,3,1,2,1,3,2,1]
    })

    payments = pd.DataFrame({
        'payment_id': range(1, 17),
        'order_id': range(1, 17),
        'payment_method': ['card', 'bank', 'card', 'card', 'bank', 'ApplePay', 'card', 'card', 'card', 'ApplePay','card','bank','card','card','bank','ApplePay'],
        'payment_date': pd.date_range('2024-03-01', periods=16, freq='7D'),
        'status': ['completed', 'completed', 'pending', 'completed', 'failed', 'completed', 'pending', 'completed', 'failed', 'completed', 'completed', 'completed', 'pending', 'completed', 'failed', 'completed'],
        'currency': ['USD', 'USD', 'USD', 'USD', 'USD', 'USD', 'USD', 'USD', 'USD', 'USD', 'USD', 'USD', 'USD', 'USD', 'USD', 'USD'],
        'amount': [10, 40, 15, 75, 60, 20, 15, 30, 30, 90, 10, 40, 15, 75, 60, 20]
    })

    reviews = pd.DataFrame({
        'review_id': range(1, 6),
        'user_id': [1,2,3,4,5],
        'product_id': [1,2,3,4,5],
        'rating': [4, 5, 3, 4, 2],
        'review_text': ['Great product!', 'Loved it!', 'It was okay.', 'Very good quality.', 'Not what I expected.'],
        'review_date': pd.date_range('2024-03-01', periods=5, freq='D')
    })

    return users, products, orders, payments, reviews

users, products, orders, payments, reviews = create_dummy_data()

# Load data into duckdb
con = duckdb.connect()
con.register('users', users)
con.register('products', products)
con.register('orders', orders)
con.register('payments', payments)
con.register('reviews', reviews)

st.title("🔍 AI SQL generator and Comparison Tool")

st.markdown("""
Write a natural language data question, and compare how two prompts generate SQL queries against it.
""")

natural_question = st.text_input("Enter your question (e.g., 'Which product had the highest total sales?')")

# Table selection dropdown and schema viewer
table_options = {
    "users": users,
    "products": products,
    "orders": orders,
    "payments": payments,
    "reviews": reviews
}
table_names = list(table_options.keys())

selected_table = st.selectbox("📂 Select a table to preview", table_names)
st.dataframe(table_options[selected_table])

# Schema display for selected table
schema_descriptions = {
    "users": "user_id, name, signup_date",
    "products": "product_id, product_name, price",
    "orders": "order_id, user_id, product_id, order_date, quantity",
    "payments": "payment_id, order_id, payment_method, amount",
    "reviews": "review_id, user_id, product_id, rating, review_text"
}
st.markdown(f"**Schema for `{selected_table}`**: {schema_descriptions[selected_table]}")

schema_hint = """
Table schemas:
- users(user_id, name, signup_date)
- products(product_id, product_name, price)
- orders(order_id, user_id, product_id, order_date, quantity)
- payments(payment_id, order_id, payment_method, amount)
- reviews(review_id, user_id, product_id, rating, review_text)
"""

prompt_a = st.text_area("✍️ Prompt A", f"""{schema_hint}

Only return a SQL query to answer:
{natural_question}
""")

prompt_b = st.text_area("✍️ Prompt B", f"""{schema_hint}

You're an expert data analyst. You are given a natural language question and two SQL queries to answer it. You are also given the schema of the tables. You must generate the query that is most appropriate for answering the question - beware of ambigous columns and other syntaxissues. The answer should be in the form of a SQL query.(only the SQL, no explanation):
{natural_question}
""")

temperature = st.slider("Model temperature", 0.0, 1.5, 0.7, step=0.1)

def generate_visualization(query_result, label):
    # Check if result is a DataFrame
    if isinstance(query_result, pd.DataFrame):
        st.write("Query Result: ", query_result.head())  # Debug: print first few rows

        # Attempt to find numeric columns for bar chart
        numeric_columns = query_result.select_dtypes(include=['number']).columns
        if len(numeric_columns) >= 2:
            st.write("Generating bar chart...")
            fig = px.bar(query_result, x=query_result.columns[0], y=numeric_columns[0], title="Bar Chart of Query Results")
            st.plotly_chart(fig,use_container_width=True, key=f"{label}_{uuid4()}")
        elif 'rating' in query_result.columns:
            st.write("Generating pie chart...")
            fig = px.pie(query_result, names='user_id', values='rating', title="Pie Chart of Ratings")
            st.plotly_chart(fig, use_container_width=True, key=f"{label}_{uuid4()}")
        else:
            st.write("Displaying the result as a table...")
            st.dataframe(query_result)
    else:
        st.error("Query result is not a DataFrame. Please check your query.")

if st.button("💬 Generate SQL Queries"):
    messages = lambda prompt: [
        {"role": "user", "content": prompt}
    ]

    response_a = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages(prompt_a),
        temperature=temperature
    )
    response_b = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages(prompt_b),
        temperature=temperature
    )

    output_a = response_a.choices[0].message.content
    output_b = response_b.choices[0].message.content

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 🧠 Prompt A Result")
        st.code(output_a, language='sql')
        try:
            result_a = con.execute(output_a).fetch_df()
            st.dataframe(result_a)
            generate_visualization(result_a,"plot_a")
        except Exception as e:
            st.error(f"Error running A's query: {e}")

    with col2:
        st.markdown("### 🤖 Prompt B Result")
        st.code(output_b, language='sql')
        try:
            result_b = con.execute(output_b).fetch_df()
            st.dataframe(result_b)
            generate_visualization(result_b,"plot_b")
        except Exception as e:
            st.error(f"Error running B's query: {e}")

    vote = st.radio("Which query was more accurate?", ["Prompt A", "Prompt B", "Neither"], key="vote")

    # Save to Firebase Firestore
    db.collection("prompt_battles").add({
        "timestamp": datetime.utcnow(),
        "question": natural_question,
        "prompt_a": prompt_a,
        "prompt_b": prompt_b,
        "response_a": output_a,
        "response_b": output_b,
        "vote": vote,
        "temperature": temperature,
        "tokens_a": {
            "prompt": response_a.usage.prompt_tokens,
            "completion": response_a.usage.completion_tokens,
            "total": response_a.usage.total_tokens
        },
        "tokens_b": {
            "prompt": response_b.usage.prompt_tokens,
            "completion": response_b.usage.completion_tokens,
            "total": response_b.usage.total_tokens
        }
    })

    st.markdown("---")
    st.markdown(f"**Prompt A Tokens:** {response_a.usage.prompt_tokens} | **Completion Tokens:** {response_a.usage.completion_tokens} | **Total:** {response_a.usage.total_tokens}")
    st.markdown(f"**Prompt B Tokens:** {response_b.usage.prompt_tokens} | **Completion Tokens:** {response_b.usage.completion_tokens} | **Total:** {response_b.usage.total_tokens}")
    st.markdown(f"**Temperature Used:** {temperature}")

st.markdown("---")
st.caption("Created by Ajith — Prompt Engineering meets SQL")
