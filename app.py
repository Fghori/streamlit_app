import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor

# Load the dataset
def load_data():
    data = pd.read_csv("data2.csv")
    
    # Fill missing values only in numeric columns
    numeric_cols = data.select_dtypes(include=['number']).columns
    data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())
    
    return data

def main():
    st.title("Beijing Air Quality Data Analysis")
    menu = ["Data Overview", "Exploratory Data Analysis (EDA)", "Modeling and Prediction"]
    choice = st.sidebar.selectbox("Select Section", menu)
    data = load_data()

    if choice == "Data Overview":
        st.subheader("Dataset Overview")
        st.write("### Dataset Information")
        st.write(data.info())
        st.write("### First Few Rows of Data")
        st.write(data.head())

        # Show summary statistics
        st.write("### Dataset Summary Statistics")
        st.write(data.describe())

    elif choice == "Exploratory Data Analysis (EDA)":
        st.subheader("Exploratory Data Analysis")

        # Year range selection (2013 - 2017)
        st.write("### Filter Data by Year Range (2013 - 2017)")
        year_range = st.slider("Select Year Range", 2013, 2017, (2013, 2017))

        # Filter data by the selected year range
        filtered_data = data[(data['year'] >= year_range[0]) & (data['year'] <= year_range[1])]

        # Show filtered data
        st.write(f"Showing Data from {year_range[0]} to {year_range[1]}")
        st.write(filtered_data)

        # PM2.5 Distribution
        st.write("### Distribution of PM2.5")
        plt.figure(figsize=(10, 6))
        sns.histplot(filtered_data['PM2.5'], kde=True, bins=50, color='blue')
        plt.title('Distribution of PM2.5')
        plt.xlabel('PM2.5 Concentration')
        plt.ylabel('Frequency')
        st.pyplot()

        # Check for missing values and handle them
        st.write("### Checking for Missing Values")
        if filtered_data[['PM2.5', 'TEMP']].isnull().any().any():
            st.write("Dataset contains missing values. Filling them with mean values.")
            filtered_data['PM2.5'].fillna(filtered_data['PM2.5'].mean(), inplace=True)
            filtered_data['TEMP'].fillna(filtered_data['TEMP'].mean(), inplace=True)
        else:
            st.write("No missing values found in PM2.5 and Temperature columns.")

        # Plotting PM2.5 vs Temperature
        st.write("### PM2.5 vs Temperature")
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=filtered_data['TEMP'], y=filtered_data['PM2.5'], color='orange', alpha=0.7, marker='x')
        plt.title("PM2.5 vs Temperature", fontsize=16)
        plt.xlabel("Temperature (°C)", fontsize=14)
        plt.ylabel("PM2.5 (µg/m³)", fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.ticklabel_format()
        st.pyplot()

        # Correlation heatmap
        st.write("### Correlation Heatmap")
        # Filter numeric columns only
        numeric_data = filtered_data.select_dtypes(include=['float64', 'int64'])

        # Compute correlation matrix
        corr_matrix = numeric_data.corr()

        # Plot heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
        plt.title('Correlation Heatmap')
        st.pyplot()

    elif choice == "Modeling and Prediction":
        st.subheader("Modeling and Prediction")

        # Define features and target variable
        features = ['PM10', 'SO2', 'NO2', 'CO', 'O3', 'TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM', 'wd']
        target = 'PM2.5'

        X = data[features]
        y = data[target]

        # Convert categorical features to numeric
        le = LabelEncoder()
        X['wd'] = le.fit_transform(X['wd'].astype(str))  # Convert 'wd' to numeric using LabelEncoder

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        # Linear Regression Model
        lr_model = LinearRegression()
        lr_model.fit(X_train, y_train)

        # Predictions
        y_pred_lr = lr_model.predict(X_test)

        # Evaluate the model
        mse_lr = mean_squared_error(y_test, y_pred_lr)
        r2_lr = r2_score(y_test, y_pred_lr)
        
        # Random Forest Model
        rf_model = RandomForestRegressor(random_state=42)
        rf_model.fit(X_train, y_train)

        # Predictions
        y_pred_rf = rf_model.predict(X_test)

        # Evaluate the model
        mse_rf = mean_squared_error(y_test, y_pred_rf)
        r2_rf = r2_score(y_test, y_pred_rf)

        # Display evaluation metrics
        st.write(f"### Linear Regression - Mean Squared Error: {mse_lr:.4f}, R-squared: {r2_lr:.4f}")
        st.write(f"### Random Forest - Mean Squared Error: {mse_rf:.4f}, R-squared: {r2_rf:.4f}")

        # Visualization of model comparison
        st.write("### Comparison of Models")

        # Bar plot for MSE and R-squared
        model_metrics = pd.DataFrame({
            'Model': ['Linear Regression', 'Random Forest'],
            'Mean Squared Error': [mse_lr, mse_rf],
            'R-squared': [r2_lr, r2_rf]
        })

        plt.figure(figsize=(10, 6))
        model_metrics.set_index('Model')[['Mean Squared Error', 'R-squared']].plot(kind='bar', color=['red', 'green'])
        plt.title('Model Comparison: MSE and R-squared')
        plt.ylabel('Score')
        plt.xlabel('Model')
        st.pyplot()

        # Scatter plot of actual vs predicted values for Linear Regression
        st.write("### Linear Regression: Actual vs Predicted Values")
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred_lr, color='blue', alpha=0.6, edgecolors='k', label='Predictions')
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Ideal Fit')
        plt.title('Linear Regression: Actual vs Predicted Values', fontsize=16)
        plt.xlabel('Actual Values (y_test)', fontsize=14)
        plt.ylabel('Predicted Values (y_pred_lr)', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(alpha=0.3)
        st.pyplot()

        # Residuals plot for Linear Regression
        residuals_lr = y_test - y_pred_lr
        st.write("### Linear Regression: Residuals Plot")
        plt.figure(figsize=(10, 6))
        plt.scatter(y_pred_lr, residuals_lr, color='purple', alpha=0.6, edgecolors='k')
        plt.axhline(y=0, color='r', linestyle='--', lw=2)
        plt.title('Linear Regression: Residuals Plot', fontsize=16)
        plt.xlabel('Predicted Values (y_pred_lr)', fontsize=14)
        plt.ylabel('Residuals (y_test - y_pred_lr)', fontsize=14)
        plt.grid(alpha=0.3)
        st.pyplot()

        # Scatter plot of actual vs predicted values for Random Forest
        st.write("### Random Forest: Actual vs Predicted Values")
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred_rf, color='blue', alpha=0.6, edgecolors='k', label='Predictions')
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Ideal Fit')
        plt.title('Random Forest: Actual vs Predicted Values', fontsize=16)
        plt.xlabel('Actual Values (y_test)', fontsize=14)
        plt.ylabel('Predicted Values (y_pred_rf)', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(alpha=0.3)
        st.pyplot()

        # Residuals plot for Random Forest
        residuals_rf = y_test - y_pred_rf
        st.write("### Random Forest: Residuals Plot")
        plt.figure(figsize=(10, 6))
        plt.scatter(y_pred_rf, residuals_rf, color='green', alpha=0.6, edgecolors='k')
        plt.axhline(y=0, color='r', linestyle='--', lw=2)
        plt.title('Random Forest: Residuals Plot', fontsize=16)
        plt.xlabel('Predicted Values (y_pred_rf)', fontsize=14)
        plt.ylabel('Residuals (y_test - y_pred_rf)', fontsize=14)
        plt.grid(alpha=0.3)
        st.pyplot()

if __name__ == "__main__":
    main()
