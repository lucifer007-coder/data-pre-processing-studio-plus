# Data Preprocessing Studio 🧹

**Data Preprocessing Studio** is a Streamlit-based web application designed to simplify and streamline data preprocessing and feature engineering tasks for data scientists and analysts. It offers an interactive, modular, and scalable interface for handling datasets, supporting both small-scale (Pandas) and large-scale (Dask) data processing. With a pipeline-based workflow, users can upload datasets, apply preprocessing steps, visualize changes, and export results with ease.

## Table of Contents 📋
- [Features](#features) ✨
- [Prerequisites](#prerequisites) ⚙️
- [Installation](#installation) 🛠️
- [Usage](#usage) 📈
- [Project Structure](#project-structure) 🗂️
- [Modules](#modules) 🧩
- [Dependencies](#dependencies) 📦
- [Configuration](#configuration) 🔧
- [Contributing](#contributing) 🤝
- [License](#license) 📜
- [Contact](#contact) 📬

## Features ✨
- **Data Upload** 📂: Import CSV files or `.dps` bundle files to start or resume sessions.
- **Preprocessing Pipeline** 🧪: Build, preview, and apply a chain of preprocessing steps with undo/redo functionality.
- **Preprocessing Tasks** 🛠️:
  - **Missing Data** 🧩: Impute (mean, median, mode, KNN, etc.) or drop missing values.
  - **Duplicates** 🗑️: Remove duplicate rows based on selected columns.
  - **Categorical Encoding** 🔢: Supports one-hot, label, ordinal, and high-cardinality encoding (target, frequency, hashing).
  - **Feature Scaling** 📏: Standard, MinMax, and robust scaling with option to keep original columns.
  - **Outliers** 📈: Detect and handle outliers using IQR or Z-score methods.
  - **Imbalanced Data** ⚖️: Rebalance classification datasets via oversampling or undersampling.
  - **Text Preprocessing** 📝: Clean text (stopwords removal, lemmatization) and extract TF-IDF features.
  - **Time-Series Preprocessing** ⏳: Smooth data (moving average, Savitzky-Golay) and resample to different frequencies.
  - **Data Inconsistency** 📏: Normalize text, standardize dates, convert units, and extract domains from URLs.
  - **Feature Engineering** 🛠️: Create polynomial, clustering, PCA, statistical, and custom features.
- **Automated Recommendations** 💡: Analyze datasets and suggest preprocessing steps for data quality issues (e.g., missing values, outliers).
- **Data Exploration** 📊: Interactive dashboard with statistics, correlations, PII detection, and visualizations (histograms, time-series plots, heatmaps).
- **Export Options** 💾: Export datasets as CSV, Parquet, Excel, Feather, or SQLite, and pipelines as JSON or Python code.
- **Scalability** 🚀: Handles large datasets using Dask, with automatic switching for files >100MB.
- **Visualization** 🎨: Integrates Altair for histograms, line plots, and word clouds, with AgGrid for enhanced table displays (optional).
- **Security** 🔒: Detects PII (e.g., emails, SSNs) and validates SQL queries to prevent code injection.

## Prerequisites ⚙️
- **Python** 🐍: Version 3.8 or higher.
- **Operating System** 💻: Windows, macOS, or Linux.
- **Dependencies** 📦: Listed in `requirements.txt` (see [Dependencies](#dependencies)).
- **Optional** 🔧: `duckdb` for SQL query explorer, `st_aggrid` for enhanced table displays.

## Installation 🛠️
1. **Clone the Repository** 📥:
   ```bash
   git clone https://github.com/your-username/data-preprocessing-studio.git
   cd data-preprocessing-studio
   ```

2. **Create a Virtual Environment** 🛡 (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies** 📦:
   ```bash
   pip install -r requirements.txt
   ```

4. **Download NLTK Data** 📚 (for text preprocessing):
   ```python
   import nltk
   nltk.download('stopwords')
   nltk.download('wordnet')
   ```

5. **Run the Application** 🚀:
   ```bash
   streamlit run app.py
   ```
   The application will open in your default web browser at `http://localhost:8501`. 🌐

#### You can try the app here: [https://data-pre-processing-studio.streamlit.app/] 

## Usage 📈
1. **Launch the Application** 🚀:
   Run `streamlit run app.py` to start the web interface.

2. **Upload a Dataset** 📂:
   - Navigate to the **Upload** section.
   - Upload a CSV file or a `.dps` bundle to resume a previous session.
   - Preview the dataset and its shape. 👀

3. **Explore and Preprocess Data** 🧹:
   - Use the **Dashboard** to view statistics, correlations, and PII. 📊
   - Apply preprocessing steps (e.g., missing data handling, encoding) in respective sections.
   - Preview changes on a sampled dataset and add steps to the pipeline. 🔍

4. **Manage the Pipeline** 🧪:
   - In the **Pipeline & Preview** section, review queued steps, preview the full pipeline, or apply it to the dataset.
   - Undo changes or clear the pipeline as needed. 🔄

5. **Export Results** 💾:
   - Export the processed dataset in multiple formats (CSV, Parquet, etc.).
   - Save the pipeline as JSON or Python code for reproducibility. 🐍
   - Export the session as a `.dps` bundle to resume later.

6. **Use Recommendations** 💡:
   - In the **Recommendations** section, review automated suggestions for data quality issues.
   - Add suggested steps to the pipeline with one click. ✅

## Project Structure 🗂️
```
data-preprocessing-studio/
├── app.py                 # Main application entry point 
├── session.py             # Session state management 🗃
├── components/
│   ├── sections/
│   │   ├── upload.py      # Data upload module 
│   │   ├── missing_data.py # Missing data handling 
│   │   ├── inconsistency.py # Data inconsistency handling 
│   │   ├── duplicates.py   # Duplicate removal
│   │   ├── encoding.py     # Categorical encoding
│   │   ├── scaling.py      # Feature scaling
│   │   ├── outliers.py     # Outlier handling 
│   │   ├── imbalanced.py   # Imbalanced data handling 
│   │   ├── pipeline_preview.py # Pipeline management 
│   │   ├── time_series.py  # Time-series preprocessing 
│   │   ├── text.py         # Text preprocessing 
│   │   ├── feature_engineering.py # Feature engineering 
│   │   ├── dashboard.py    # Data exploration dashboard
│   │   ├── export.py       # Data and pipeline export
│   │   ├── recommendations.py # Automated preprocessing recommendations
│   ├── sidebar.py         # Sidebar navigation
├── utils/
│   ├── data_utils.py      # Data processing utilities 
│   ├── stats_utils.py     # Statistical analysis utilities
│   ├── viz_utils.py       # Visualization utilities
│   ├── bundle_io.py       # Bundle import/export utilities 
├── preprocessing/
│   ├── pipeline.py        # Pipeline execution logic
│   ├── steps.py           # Preprocessing step implementations 
├── requirements.txt       # Project dependencies
├── README.md              # This file 
```

## Modules 🧩
- **app.py**: Configures the Streamlit app and orchestrates section rendering. 🚀
- **session.py**: Manages session state, including dataset, pipeline, history, and changelog. 🗃️
- **upload.py**: Handles CSV and `.dps` bundle uploads with preview. 📂
- **missing_data.py**: Imputes or drops missing values with strategies like mean, median, KNN, etc. 🧩
- **inconsistency.py**: Normalizes text, standardizes dates, converts units, and extracts URL domains. 📏
- **duplicates.py**: Removes duplicate rows based on selected columns. 🗑️
- **encoding.py**: Encodes categorical variables (one-hot, label, ordinal, high-cardinality). 🔢
- **scaling.py**: Scales numeric features (standard, MinMax, robust). 📏
- **outliers.py**: Detects and handles outliers using IQR or Z-score. 📈
- **imbalanced.py**: Rebalances classification datasets via oversampling or undersampling. ⚖️
- **pipeline_preview.py**: Manages pipeline execution, preview, and bundle export. 🧪
- **time_series.py**: Smooths and resamples time-series data. ⏳
- **text.py**: Cleans text and extracts TF-IDF features. 📝
- **feature_engineering.py**: Creates and selects advanced features (polynomial, PCA, clustering). 🛠️
- **dashboard.py**: Provides data exploration with statistics, visualizations, and SQL queries. 📊
- **export.py**: Exports datasets and pipelines in multiple formats. 💾
- **recommendations.py**: Analyzes datasets and suggests preprocessing steps. 💡

## Dependencies 📦
The project relies on the following Python libraries (see `requirements.txt` for versions):
- **Core** 🧠: `streamlit`, `pandas`, `numpy`, `dask[distributed]`, `pyarrow`
- **Machine Learning** 🤖: `scikit-learn`, `dask_ml`, `featuretools`
- **Visualization** 🎨: `altair`, `matplotlib`, `wordcloud`, `pillow`
- **Text Processing** 📝: `nltk`, `regex`
- **Database and Export** 💾: `openpyxl`, `sqlparse`, `duckdb`, `pyyaml`, `joblib`
- **Utilities** 🛠️: `setuptools`, `numexpr`

Install dependencies using:
```bash
pip install -r requirements.txt
```

## Configuration 🔧
- **Session State** 🗃️: Managed in `session.py`, with variables for dataset (`df`, `raw_df`), pipeline, history, and changelog.
- **Large Datasets** 📈: Automatically uses Dask for files >100MB, with a block size of 64MB.
- **Visualization** 🎨: Configurable chart dimensions (width: 400px, height: 300px) in `dashboard.py` and `pipeline_preview.py`.
- **Feature Engineering** 🛠️: Configurable parameters (e.g., max polynomial degree, correlation threshold) in `feature_engineering.py`.
- **Security** 🔒: PII detection and SQL query validation in `dashboard.py` and `export.py`.

To customize configurations, modify constants in the respective modules (e.g., `CONFIG` in `dashboard.py`).

## License 📜
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact 📬
For questions, bug reports, or feature requests, please open an issue on GitHub or contact me at [ketanedumail@gmail.com]. 📧

*Built with ❤️ using Streamlit.*
