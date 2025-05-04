# AI SQL Generator and Comparison Tool

The AI SQL Generator and Comparison Tool is a web application built using Streamlit that helps users generate SQL queries based on natural language questions. The tool compares the generated SQL queries using two different prompts and displays the results alongside visualizations for easy comparison. It's an ideal tool for data analysts, developers, and anyone working with databases.

## 🚀 Features

- 🔍 **Natural Language to SQL** — Generate SQL queries from questions like _“Which product had the highest total sales?”_
- 🧪 **Prompt A vs. Prompt B Comparison** — Easily evaluate which prompt generates more accurate or readable queries.
- 📊 **Table + Chart Output** — View results and compare query outputs visually.
- ☁️ **Firebase Integration** — Logs user queries, votes, and model usage stats for later analysis.
- 💬 **User Feedback** — Vote on which prompt generated a better result.

---

## 📦 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/ai-sql-generator.git
   cd ai-sql-generator
   ``` 
2. **Create and activate a virtual environment**
   ```bash
    python3 -m venv venv
    source venv/bin/activate     # macOS/Linux
    venv\Scripts\activate        # Windows
   ```
3. **Install dependencies**
   ```bash
    pip install -r requirements.txt
   ```
4. **Set up environment variables**
Create a .env file in the root directory:
   ```bash
    OPENAI_API_KEY=your-openai-api-key
   ```
5. **Add Firebase credentials**

Download your Firebase service account key.

Save it as: firebase_service_account.json in the project root.

## ▶️ Running the App
 ```bash
  streamlit run app.py
```
🧠 Example Workflow
Select a table to view its schema.

Enter a natural language data question.

Submit two prompt styles to generate SQL.

Compare outputs + results.

Vote on the better SQL query.

## 📷 Screenshots

<img width="867" alt="image" src="https://github.com/user-attachments/assets/bde60dc8-c45a-4574-b43b-211fe9009811" />
<img width="945" alt="image" src="https://github.com/user-attachments/assets/cfab876b-8e98-4cc0-b300-19ed53efaec7" />
<img width="958" alt="image" src="https://github.com/user-attachments/assets/33151014-3d8e-4709-b1ed-1986dfd47c69" />





