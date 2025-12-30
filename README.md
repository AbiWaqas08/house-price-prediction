House Price Prediction using Machine Learning & FastAPI

Project Overview
This project implements an "end-to-end Machine Learning pipeline" to predict house prices based on numerical features such as house size, number of rooms, age of the house, and distance from the city center.  
The final trained model is deployed using "FastAPI", providing a REST API for real-time house price prediction.

This project is suitable for:
- AI / ML internships
- Data Science portfolios
- Demonstrating regression modeling and API deployment skills

---

Problem Statement
House price prediction is a common real-world regression problem in the real estate domain.  
The objective of this project is to predict the "price of a house" using the following features:

- Square footage of the house
- Number of rooms
- Age of the house
- Distance from the city center

This is a "supervised learning regression problem".

---

Dataset Description
The dataset consists of "10,000 records" and "5 numerical columns".

| Column Name | Description |
|------------|------------|
| `square_feet` | Total area of the house |
| `num_rooms` | Number of rooms |
| `age` | Age of the house (years) |
| `distance_to_city` | Distance from city center (km) |
| `price` | House price (Target variable) |

Dataset Characteristics
- No missing values
- No categorical features
- Clean and well-structured
- Ideal for regression models

---

Exploratory Data Analysis (EDA)
The following EDA steps were performed:

- Checking dataset shape and column information
- Statistical summary using "describe()"
- Analysis of feature distributions
- Detection of outliers using the "IQR (Interquartile Range) method"

Outlier Detection & Removal
- Outliers were identified using the IQR method
- Approximately "1–2%" extreme values were removed
- Outlier removal improved model stability and reduced prediction error

---

Data Preprocessing
- No encoding required (all features are numeric)
- Outliers removed using IQR method
- Features and target variable separated
- Dataset split into "80% training" and "20% testing" data

---

Models Trained
The following regression models were trained and evaluated:

1. Linear Regression ✅
2. Decision Tree Regressor
3. Random Forest Regressor

---

Model Evaluation
Models were evaluated using standard regression metrics:

- R² Score
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)

Model Performance Comparison

| Model | R² Score | MAE |
|------|---------|-----|
| Linear Regression | 0.958 | 15,918 |
| Random Forest | 0.947 | 18,004 |
| Decision Tree | 0.890 | 25,652 |

Best Model Selection
"Linear Regression" was selected as the final model because:
- It achieved the highest R² score
- It produced the lowest prediction error
- The dataset showed a strong linear relationship
- The model is simple, stable, and interpretable

---

Model Deployment using FastAPI
The trained Linear Regression model was deployed using "FastAPI", allowing real-time predictions through HTTP requests.

-------API Endpoints------

Health Check

GET /


Predict House Price

POST /predict

---

Sample API Request
json
{
  "square_feet": 2000,
  "num_rooms": 4,
  "age": 5,
  "distance_to_city": 10
}


### 📤 Sample API Response

```json
{
  "predicted_price": 245320.75
}
```

---

## ▶ How to Run the Project Locally

### Step 1: Clone the Repository

```bash
git clone https://github.com/AbiWaqas08/house-price-prediction-ml.git
cd house-price-prediction-ml
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Run FastAPI Server

```bash
uvicorn main:app --reload
```

### Step 4: Open Swagger UI

Open your browser and go to:

```
http://127.0.0.1:8000/docs
```

---

Tech Stack
* Python
* Pandas
* NumPy
* Scikit-learn
* FastAPI
* Uvicorn
* Joblib
* Git & GitHub

---

Project Structure
house-price-prediction/
│
├── main.py
├── model/
│   └── house_price_model.pkl
├── requirements.txt
├── README.md
└── .gitignore


---

Key Learnings
* End-to-end regression pipeline development
* Practical outlier detection and removal
* Model comparison and selection
* Deploying ML models using FastAPI
* Structuring ML projects for GitHub and internships

---

Future Enhancements
* Add feature scaling comparison
* Integrate frontend using React
* Deploy API to cloud platforms (Render / AWS)
* Dockerize the application

---

Abi Waqas
AI / ML Engineer
Pakistan

