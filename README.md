
UPDATED VERSION IS ON BRANCH V2
```markdown
# 🧠 Invoice Anomaly Detection App

A simple **AI-powered anomaly detection** web application built with **Streamlit** and **Scikit-learn (Isolation Forest)**.  
The app detects unusual invoice entries using your trained model and provides real-time predictions via an interactive web interface.

---

## 🚀 Features

- 🔍 Detects anomalies in invoice data using **Isolation Forest**
- 🧹 Automatic data preprocessing and training
- 💾 Saves trained model, encoders, and column metadata
- 🌐 Clean and interactive **Streamlit UI**
- ⚡ Automatically trains only once if model files are missing

---

## 🧩 Project Structure

```

invoice-anomaly-app/
│
├── app.py                     # Main Streamlit application
├── model.pkl                  # Trained Isolation Forest model (auto-generated)
├── encoders.pkl               # Encoded categorical data mappings (auto-generated)
├── columns.pkl                # List of columns used for training (auto-generated)
├── data.csv                   # Dataset file
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation

````

---

## 🧰 Requirements

Ensure you have **Python 3.8+** installed.

### Install dependencies

```bash
pip install -r requirements.txt
````

Or manually install **Streamlit** (if not already included):

```bash
pip install streamlit
```

---

## ⚙️ How It Works

1. The app checks if model files (`model.pkl`, `encoders.pkl`, `columns.pkl`) exist.

   * If not, it **preprocesses** your dataset and **trains** a new model.
2. The trained model, encoders, and column list are saved locally.
3. When the app runs, it loads these files and displays a form in the UI.
4. You enter invoice details → The app encodes the input → Runs anomaly detection.
5. It displays whether your input is **Normal** or **Anomalous**.

---

## ▶️ Run the App

You can start the app in either of these two ways:

### Option 1 — Using Python

```bash
python app.py
```

### Option 2 — Using Streamlit

```bash
streamlit run app.py
```

After running, Streamlit will launch the web app at:
👉 `http://localhost:8501`

---

## 🧠 Example Workflow

1. Launch the app with `streamlit run app.py`.
2. Fill in invoice details in the input form.
3. Click **Submit**.
4. The app shows:

   * ✅ **Normal** → Data is consistent with learned patterns.
   * ⚠️ **Anomalous** → Detected deviation or unusual pattern.

---

## 📦 Model Files

These files are generated automatically during training:

| File           | Description                               |
| -------------- | ----------------------------------------- |
| `model.pkl`    | Trained Isolation Forest model            |
| `encoders.pkl` | Label encoders for categorical columns    |
| `columns.pkl`  | List of dataset columns used for training |

---

## 🧾 Example `requirements.txt`

If you don’t have a requirements file yet, you can create one with the following content:

```
pandas
numpy
scikit-learn
streamlit
joblib
```

Install them using:

```bash
pip install -r requirements.txt
```

---

## 💡 Example Code Snippet (app.py)

Below is a **minimal structure** for your `app.py` file if you want to verify your setup:

```python
import streamlit as st
import pandas as pd
import joblib
import os
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder

# ---- Preprocessing Function ----
def preprocess_data(df):
    for col in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
    return df

# ---- Train and Save Model ----
def train_and_save_model(data_path='data.csv'):
    df = pd.read_csv(data_path)
    df = preprocess_data(df)

    model = IsolationForest(contamination=0.05, random_state=42)
    model.fit(df)

    joblib.dump(model, 'model.pkl')
    joblib.dump(list(df.columns), 'columns.pkl')
    return model

# ---- Load or Train ----
def load_or_train_model():
    if os.path.exists('model.pkl') and os.path.exists('columns.pkl'):
        model = joblib.load('model.pkl')
        columns = joblib.load('columns.pkl')
    else:
        model = train_and_save_model()
        columns = joblib.load('columns.pkl')
    return model, columns

# ---- Streamlit App ----
st.title("🧠 Invoice Anomaly Detection")

model, columns = load_or_train_model()

# Input Form
with st.form("input_form"):
    st.subheader("Enter Invoice Details")
    inputs = {col: st.text_input(col) for col in columns}
    submitted = st.form_submit_button("Submit")

    if submitted:
        input_df = pd.DataFrame([inputs])
        input_df = preprocess_data(input_df)
        prediction = model.predict(input_df)[0]
        result = "✅ Normal" if prediction == 1 else "⚠️ Anomalous"
        st.success(f"Result: {result}")
```

---

## 💡 Future Enhancements

* 📊 Add anomaly score visualizations
* 📁 Support batch uploads for multi-record detection
* 💾 Integrate database logging for results
* 🌐 Deploy using **Streamlit Cloud** or **AWS EC2**

---

