# 📊 Smart EDA Analyzer

> **Upload any dataset. Understand everything about it — instantly.**

Smart EDA Analyzer is an open-source, AI-powered Exploratory Data Analysis web app built with Python and Streamlit. Drop in any CSV or Excel file and get instant statistical summaries, beautiful interactive charts, data cleaning tools, and AI-generated insights — no coding required.

---

## ✨ Features

| Feature                      | Description                                                                    |
|-----------------------------|----------------------------------------------------------------------------------|
| 📁 **Universal Upload**     | Supports CSV and Excel files up to 200MB, with auto encoding detection          |
| 🔍 **Instant Overview**     | Row/column count, missing values, duplicates, data types at a glance            |
| 💡 **Auto Insights**        | Automatically detects skewness, outliers, correlations, and data quality issues |
| 🧹 **Data Cleaning**        | Fill missing values, drop duplicates, remove outliers, rename/drop columns      |
| 📈 **Distributions**        | Histograms, box plots, bar charts, pie charts for every column                  |
| 📅 **Time Series**          | Auto-detects date columns and plots trends over time                            |
| 🔗 **Correlations**         | Interactive heatmap + ranked correlation pairs table                            |
| 🔎 **Scatter & Pair Plots** | Custom scatter plot builder + full pair plot matrix                             |
| 🚨 **Outlier Detection**    | IQR-based outlier detection with one-click removal                              |
| 🤖 **AI Insights**          | Gemini AI analyzes your dataset and returns expert observations                 |
| 💬 **Chat with Data**       | Ask questions about your dataset in natural language                            |
| 📥 **Export Report**        | Download a full HTML analysis report with one click                             |
| 🎨 **Theme Selector**       | 5 color themes: Purple, Blue, Green, Red, Orange                                |

---

## 🖥️ Demo

![EDA Analyzer Screenshot](https://img.icons8.com/fluency/200/combo-chart.png)

> Upload → Explore → Clean → Export → Done.

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- A free [Google AI Studio](https://aistudio.google.com) API key (for AI features)

### Installation

**1. Clone the repository**
```bash
git clone https://github.com/itsMouna/smart-eda-analyzer.git
cd smart-eda-analyzer
```

**2. Create and activate a virtual environment**
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Set up your API key**

Create a file at `.streamlit/secrets.toml`:
```toml
GEMINI_API_KEY = "your-gemini-api-key-here"
```

Get your free key at [aistudio.google.com](https://aistudio.google.com) → Get API Key.

**5. Run the app**
```bash
python -m streamlit run app.py
```

Open your browser at `http://localhost:8501` 🎉

---

## 📦 Tech Stack

| Tool                                         | Purpose                    |
|----------------------------------------------|----------------------------|
| [Streamlit](https://streamlit.io)            | Web app framework          |
| [Pandas](https://pandas.pydata.org)          | Data manipulation          |
| [Plotly](https://plotly.com)                 | Interactive visualizations |
| [Google Gemini](https://aistudio.google.com) | AI-powered insights        |
| [Statsmodels](https://www.statsmodels.org)   | Statistical analysis       |

---

## 📁 Project Structure

```
smart-eda-analyzer/
├── app.py                  # Main application
├── requirements.txt        # Python dependencies
├── .streamlit/
│   ├── config.toml         # Theme & server config
│   └── secrets.toml        # API keys (never committed)
├── .gitignore
└── README.md
```

---

## 🔒 Security

- API keys are stored in `.streamlit/secrets.toml` which is listed in `.gitignore` and **never pushed to GitHub**
- No user data is stored or logged anywhere
- All analysis happens locally in your session

---

## 🤝 Contributing

Contributions are welcome! Feel free to open an issue or submit a pull request.

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

---

## 👤 Author

Built by **[Mouna Ben Amor]** — feel free to connect on [LinkedIn]([https://www.linkedin.com/in/mouna-ben-amor-435a33363/?skipRedirect=true]) or [GitHub]([https://github.com/itsMouna])

---

<p align="center">Made with ❤️ and Python</p>
