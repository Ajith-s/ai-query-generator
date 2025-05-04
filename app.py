import streamlit as st
import pandas as pd
import duckdb
from openai import OpenAI
import os
from dotenv import load_dotenv
import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime
import plotly.express as px
from uuid import uuid4

# Load environment variables
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Firebase initialization
if not firebase_admin._apps:
    cred_path = os.path.join(os.path.dirname(__file__), "firebase_service_account.json")
    cred = credentials.Certificate(cred_path)
    firebase_admin.initialize_app(cred)
db = firestore.client()

# IP-based rate limiting (max 5/day)
ip = st.query_params.get("ip", [str(uuid4())])[0]
ip_doc = db.collection("rate_limits").document(ip)
ip_data = ip_doc.get().to_dict() or {}

now = datetime.utcnow()
today = now.strftime("%Y-%m-%d")
if ip_data and ip_data.get("date") == today and ip_data.get("count", 0) >= 5:
    st.error("You have reached the maximum of 5 queries today. Please come back tomorrow.")
    st.stop()

# Create synthetic data
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
        'order_date': pd.date_range('2024-03-01', periods=16, freq='7D'),
        'status': ['completed', 'completed', 'pending', 'completed', 'failed', 'completed', 'pending', 'completed', 'failed', 'completed', 'completed', 'completed', 'pending', 'completed', 'failed', 'completed'],
        'quantity': [1,2,1,3,2,1,1,2,1,3,1,2,1,3,2,1]
    })

    payments = pd.DataFrame({
        'payment_id': range(1, 17),
        'order_id': range(1, 17),
        'payment_method': ['card', 'bank', 'card', 'card', 'bank', 'ApplePay', 'card', 'card', 'card', 'ApplePay','card','bank','card','card','bank','ApplePay'],
        'payment_date': pd.date_range('2024-03-01', periods=16, freq='7D'),
        'status': ['completed', 'completed', 'pending', 'completed', 'failed', 'completed', 'pending', 'completed', 'failed', 'completed', 'completed', 'completed', 'pending', 'completed', 'failed', 'completed'],
        'currency': ['USD'] * 16,
        'amount': [10, 40, 15, 75, 60, 20, 15, 30, 30, 90, 10, 40, 15, 75, 60, 20]
    })

    reviews = pd.DataFrame({
        'review_id': list(range(1, 31)),
        'user_id': [1,2,3,4,5]*6,
        'product_id': [1,2,3,4,5]*6,
        'rating': [4, 5, 3, 4, 2, 5, 3, 4, 2, 5, 2, 4, 5, 3, 4, 3, 5, 2, 4, 3, 5, 3, 2, 4, 5, 4, 3, 2, 5, 4],
        'review_text': ['Great product!', 'Loved it!', 'It was okay.', 'Very good quality.', 'Not what I expected.']*6,
        'review_date': pd.date_range('2024-03-01', periods=30, freq='D')
    })

    return users, products, orders, payments, reviews

users, products, orders, payments, reviews = create_dummy_data()

# Load into DuckDB
con = duckdb.connect()
con.register('users', users)
con.register('products', products)
con.register('orders', orders)
con.register('payments', payments)
con.register('reviews', reviews)

st.title("🔍 AI SQL Generator and Comparison Tool")
st.markdown("""
Write a natural language data question, and compare how two prompts generate SQL queries against it.
""")

natural_question = st.text_input("Enter your question (e.g., 'Which product had the highest total sales?')")

# Table preview
if 'shown_table' not in st.session_state:
    st.session_state.shown_table = False

table_options = {
    "users": users,
    "products": products,
    "orders": orders,
    "payments": payments,
    "reviews": reviews
}
table_names = list(table_options.keys())
selected_table = st.selectbox("📂 Select a table to preview", table_names)
if not st.session_state.shown_table:
    st.dataframe(table_options[selected_table])
    st.session_state.shown_table = True

# Schema hint (used internally only)
schema_hint = """
Table schemas:
- users(user_id, name, signup_date)
- products(product_id, product_name, price)
- orders(order_id, user_id, product_id, order_date, quantity)
- payments(payment_id, order_id, payment_method, amount)
- reviews(review_id, user_id, product_id, rating, review_text)
""".strip()

prompt_a = st.text_area("✍️ Prompt A", f"""{schema_hint}

Only return a SQL query to answer:
{natural_question}
""")

prompt_b = st.text_area("✍️ Prompt B", f"""{schema_hint}

You're an expert data analyst. You are given a natural language question and two SQL queries to answer it. You are also given the schema of the tables. You must generate the query that is most appropriate for answering the question - beware of ambiguous columns and other syntax issues. The answer should be in the form of a SQL query (only the SQL, no explanation):
{natural_question}
""")

temperature = st.slider("Model temperature", 0.0, 1.5, 0.7, step=0.1)

def generate_visualization(df, label):
    if not isinstance(df, pd.DataFrame):
        st.error("Query result is not a DataFrame.")
        return

    st.write("Query Result:", df.head())
    numeric_columns = df.select_dtypes(include=['number']).columns
    if len(numeric_columns) >= 2:
        fig = px.bar(df, x=df.columns[0], y=numeric_columns[0], title="Bar Chart")
        st.plotly_chart(fig, use_container_width=True, key=f"{label}_{uuid4()}")
    elif 'rating' in df.columns:
        fig = px.pie(df, names='user_id', values='rating', title="Pie Chart of Ratings")
        st.plotly_chart(fig, use_container_width=True, key=f"{label}_{uuid4()}")
    else:
        st.dataframe(df)

if st.button("💬 Generate SQL Queries"):
    def messages(prompt):
        return [{"role": "user", "content": prompt}]

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
    with col2:
        st.markdown("### 🤖 Prompt B Result")
        st.code(output_b, language='sql')

    try:
        result_a = con.execute(output_a).fetch_df()
        result_b = con.execute(output_b).fetch_df()

        if result_a.equals(result_b):
            st.markdown("### ✅ Both queries returned the same result:")
            st.dataframe(result_a)
            generate_visualization(result_a, "plot")
        else:
            st.markdown("### 🔍 Different Results")
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Prompt A")
                st.dataframe(result_a)
                generate_visualization(result_a, "plot_a")
            with col2:
                st.subheader("Prompt B")
                st.dataframe(result_b)
                generate_visualization(result_b, "plot_b")
    except Exception as e:
        st.error(f"Error executing query: {e}")

    vote = st.radio("Which query was more accurate?", ["Prompt A", "Prompt B", "Neither", "Both"], key="vote")

    db.collection("prompt_battles").add({
        "timestamp": datetime.utcnow(),
        "question": natural_question,
        "prompt_a": prompt_a,
        "prompt_b": prompt_b,
        "response_a": output_a,
        "response_b": output_b,
        "vote": vote,
        "temperature": temperature,
        "tokens_a": response_a.usage.model_dump() if hasattr(response_a, 'usage') else {},
        "tokens_b": response_b.usage.model_dump() if hasattr(response_b, 'usage') else {}
    })

    ip_doc.set({"count": ip_data.get("count", 0) + 1, "date": today})

    st.markdown("---")
    st.markdown(f"**Prompt A Tokens:** {response_a.usage.total_tokens if hasattr(response_a, 'usage') else 'N/A'}")
    st.markdown(f"**Prompt B Tokens:** {response_b.usage.total_tokens if hasattr(response_b, 'usage') else 'N/A'}")
    st.markdown(f"**Temperature Used:** {temperature}")

st.markdown("---")
st.caption("Created by Ajith — Prompt Engineering meets SQL")
