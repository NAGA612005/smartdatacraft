# app.py
import streamlit as st
import pandas as pd
import re
import unidecode
import json
from io import StringIO
from bs4 import BeautifulSoup
import pdfplumber
import seaborn as sns
import matplotlib.pyplot as plt
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.memory import ConversationBufferMemory
import os
import tempfile
import asyncio

# ------------------- CLEANING UTILS -------------------
def clean_text(val):
    if not isinstance(val, str):
        return val
    val = unidecode.unidecode(val)
    val = re.sub(r"\s+", " ", val)
    val = re.sub(r"[^\w\s@.:/-]", "", val)
    return val.strip().lower()

def clean_dataframe(df):
    df = df.drop_duplicates()
    df.columns = [clean_text(c) for c in df.columns]
    for col in df.select_dtypes(include='object'):
        df[col] = df[col].astype(str).apply(clean_text)
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].replace({'': None, 'nan': None, 'none': None})
    return df

# ------------------- FILE LOADER -------------------
def load_file(file):
    name = file.name.lower()
    if name.endswith('.csv'):
        return pd.read_csv(file)
    elif name.endswith('.xlsx'):
        return pd.read_excel(file)
    elif name.endswith('.json'):
        return pd.json_normalize(json.load(file))
    elif name.endswith('.txt'):
        lines = file.read().decode("utf-8").splitlines()
        return pd.DataFrame(lines, columns=["raw"])
    elif name.endswith('.html') or name.endswith('.xml'):
        soup = BeautifulSoup(file.read(), "html.parser")
        texts = [tag.get_text() for tag in soup.find_all()]
        return pd.DataFrame(texts, columns=["raw"])
    elif name.endswith('.pdf'):
        with pdfplumber.open(file) as pdf:
            text = "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())
        return pd.DataFrame(text.splitlines(), columns=["raw"])
    else:
        raise Exception("Unsupported file type")

# ------------------- RAG USING GEMINI -------------------
def build_gemini_qa_tool(file_path, api_key):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())

    loader = TextLoader(file_path)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = splitter.split_documents(documents)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    vectordb = FAISS.from_documents(texts, embeddings)
    retriever = vectordb.as_retriever()

    gemini = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key="AIzaSyC2Z9xqIOx4BR4sjCX0Bt1sHYDZNGquMng", temperature=0.5)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    rag_chain = RetrievalQA.from_chain_type(llm=gemini, retriever=retriever, memory=memory)
    return rag_chain

# ------------------- STREAMLIT UI -------------------
st.set_page_config(page_title="SmartDataCraft ", layout="wide")
st.title("SmartDataCraft ")

file = st.file_uploader("\U0001F4C1 Upload any unstructured data file", type=["csv", "xlsx", "json", "txt", "pdf", "html", "xml"])

if file:
    try:
        df = load_file(file)
        st.subheader("\U0001F4C4 Raw Data")
        st.dataframe(df.head())

        cols_with_null = [col for col in df.columns if df[col].isnull().any()]
        for col in cols_with_null:
            st.markdown(f"### \U0001F6E0️ Handle Missing Values in `{col}`")
            method = st.selectbox(f"Choose method for `{col}`:", ["Drop", "Mean", "Median", "Mode", "Custom"], key=col)
            if method == "Drop":
                df = df[df[col].notnull()]
            elif method == "Mean" and pd.api.types.is_numeric_dtype(df[col]):
                df[col].fillna(df[col].mean(), inplace=True)
            elif method == "Median" and pd.api.types.is_numeric_dtype(df[col]):
                df[col].fillna(df[col].median(), inplace=True)
            elif method == "Mode":
                df[col].fillna(df[col].mode()[0], inplace=True)
            elif method == "Custom":
                custom_val = st.text_input(f"Enter custom value for `{col}`:", key=f"custom_{col}")
                if custom_val:
                    df[col].fillna(custom_val, inplace=True)

        cleaned = clean_dataframe(df)
        st.subheader("\U0001F9FD Cleaned Data")
        st.dataframe(cleaned.head())

        st.download_button("\U0001F4E5 Download Cleaned CSV", cleaned.to_csv(index=False), "cleaned.csv", "text/csv")

        # ----------- RAG AI CHAT -------------
        if st.checkbox("\U0001F916 Chat with the raw file using Gemini RAG"):
            gemini_key = "AIzaSyC2Z9xqIOx4BR4sjCX0Bt1sHYDZNGquMng"
            if gemini_key:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as temp_file:
                    raw_text = "\n".join(df.astype(str).apply(lambda row: " ".join(row), axis=1))
                    temp_file.write(raw_text.encode("utf-8"))
                    temp_file_path = temp_file.name

                qa_chain = build_gemini_qa_tool(temp_file_path, gemini_key)

                query = st.text_area("Ask questions about your file:")
                if st.button("Submit Query") and query:
                    result = qa_chain.run(query)
                    st.success(result)

        # ----------- ADVANCED VISUALIZATION -------------
        if st.checkbox("\U0001F4CA Data Visualization"):
            chart_type = st.selectbox("Choose Chart Type", [
                "Correlation Heatmap",
                "Bar Chart",
                "Line Chart",
                "Pie Chart",
                "Box Plot",
                "Histogram"
            ])

            numeric_df = cleaned.select_dtypes(include='number')
            categorical_cols = cleaned.select_dtypes(include='object').columns.tolist()

            if chart_type == "Correlation Heatmap" and not numeric_df.empty:
                st.subheader("Correlation Heatmap")
                fig = plt.figure(figsize=(10, 6))
                sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")
                st.pyplot(fig)

            elif chart_type in ["Bar Chart", "Line Chart"]:
                col = st.selectbox("Select column to visualize", cleaned.columns)
                chart_data = cleaned[col].value_counts().reset_index()
                chart_data.columns = [col, "count"]

                if chart_type == "Bar Chart":
                    st.bar_chart(chart_data.set_index(col))
                else:
                    st.line_chart(chart_data.set_index(col))

            elif chart_type == "Pie Chart":
                col = st.selectbox("Select column for Pie Chart", categorical_cols)
                pie_data = cleaned[col].value_counts().head(10)
                fig, ax = plt.subplots()
                ax.pie(pie_data, labels=pie_data.index, autopct='%1.1f%%')
                ax.axis('equal')
                st.pyplot(fig)

            elif chart_type == "Box Plot":
                num_col = st.selectbox("Select numeric column", numeric_df.columns)
                fig, ax = plt.subplots()
                sns.boxplot(data=cleaned, y=num_col, ax=ax)
                st.pyplot(fig)

            elif chart_type == "Histogram":
                num_col = st.selectbox("Select numeric column", numeric_df.columns)
                fig, ax = plt.subplots()
                sns.histplot(data=cleaned, x=num_col, bins=20, ax=ax)
                st.pyplot(fig)

    except Exception as e:
        st.error(f"❌ {e}")
