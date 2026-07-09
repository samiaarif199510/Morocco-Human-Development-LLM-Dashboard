# Morocco Human Development LLM Dashboard

## Project Description
This project analyses the UN report **"50 Years of Human Development & Perspectives to 2025"** for Morocco using local LLMs.

The project includes PDF processing, structured data extraction, LLM-based evaluation, and an interactive Streamlit dashboard for exploring human development indicators.

## Project Files

- `Morocco-Human-Development-LLM-Dashboard-Report.pdf`  
  Final project report.

- `Morocco_Human_Development_LLM_Pipeline.ipynb`  
  Main notebook containing the complete analysis pipeline.

- `output/`  
  Contains generated structured JSON results:
  - `task1_summaries_and_evaluation.json`  
    Report summaries and evaluation results.
  - `task2_thematic_and_numeric.json`  
    Thematic analysis and numerical indicators.

- `app.py`  
  Streamlit dashboard application.

## Technologies Used
- Python
- Streamlit
- Plotly
- Ollama (Local LLMs)

## How to Run

Install required libraries:

```bash
pip install -r requirements.txt
