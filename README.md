# 🚀 SmartDataCraft

SmartDataCraft is a powerful, Streamlit-based AI tool designed for anyone who wants to quickly upload, clean, analyze, visualize, and chat with unstructured data using Google's Gemini RAG (Retrieval-Augmented Generation).
Live link: smartdatacraft.streamlit.app
---

## ✨ Features

- 🔍 Auto-detects and cleans uploaded data (CSV, Excel, JSON, PDF, TXT, XML, HTML)
- 🛠️ Interactive UI for handling missing values (drop, mean, median, mode, custom)
- 📊 Data visualizations (Heatmap, Histogram, Pie, Box, Line, Bar)
- 🤖 Chat with your data using Gemini RAG (via LangChain)
- ⬇️ Export cleaned dataset instantly

---

## 🧰 Tech Stack

- Python + Streamlit
- Pandas, Seaborn, Matplotlib
- BeautifulSoup, pdfplumber, unidecode
- LangChain, FAISS, Gemini API (via `langchain-google-genai`)

---

## 🔧 Installation

```bash
git clone https://github.com/NAGA612005/smartdatacraft.git
cd smartdatacraft
pip install -r requirements.txt

