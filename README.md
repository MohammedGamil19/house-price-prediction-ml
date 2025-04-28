# House Price Prediction - Machine Learning Project

## 📚 Overview
This project aims to predict median house prices in California using a supervised machine learning approach (Linear Regression).  
We use the California Housing dataset from Scikit-Learn, perform data exploration, feature scaling, model training, evaluation, and save the trained model for future use.

This project demonstrates the complete end-to-end machine learning workflow including data preprocessing, visualization, model building, and performance evaluation.

---

## 🛠️ Technologies Used
- **Python**
- **Pandas** - for data handling
- **Numpy** - for numerical operations
- **Matplotlib & Seaborn** - for data visualization
- **Scikit-learn** - for machine learning models and preprocessing

---

## 📊 Dataset
- **Source:** Scikit-Learn built-in California Housing Dataset
- **Features:**
  - MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude
- **Target:** Median House Value (`MedHouseVal`)

---

## 📈 Machine Learning Workflow
1. **Import Libraries**
2. **Load the Dataset**
3. **Data Exploration** (summary statistics, checking feature distributions)
4. **Data Visualization** (target variable distribution, scatter plots)
5. **Data Splitting** (Train/Test Split)
6. **Feature Scaling** (StandardScaler)
7. **Model Training** (Linear Regression)
8. **Model Evaluation** (MSE, R² Score)
9. **Predictions Visualization** (Actual vs Predicted plot)
10. **Model Saving** (using `joblib`)

---

## ⚙️ Project Structure


---

## 🔥 Results
- **Model Performance:**  
  - Mean Squared Error (MSE): 0.5559
  - R² Score: 0.5758

The Linear Regression model performed reasonably well for a simple approach. Future improvements could include using more complex models like Decision Trees, Random Forests, or Gradient Boosting.

---

## 🚀 How to Run the Project
1. Clone this repository.
2. Open `house_price_prediction.ipynb` in Google Colab or Jupyter Notebook.
3. Install necessary libraries (Pandas, Numpy, Scikit-learn, Seaborn, Matplotlib).
4. Run all the cells.
5. Optionally, you can load the saved `.pkl` model for quick predictions.

---

## 🤔 Future Improvements
- Experiment with other models: Decision Tree Regressor, Random Forest, XGBoost.
- Perform hyperparameter tuning using GridSearchCV.
- Add more feature engineering techniques.
- Deploy the model as a web application using Flask or Streamlit.

---

## 👨‍💻 Author
- Mohammed AL-Shujaa


---

## 📜 License
This project is licensed under the [MIT License](LICENSE).

---
