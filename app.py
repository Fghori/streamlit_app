import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


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
        # data['Date'] = pd.to_datetime(data['Date'])
        # data['Year'] = data['Date'].dt.year
        filtered_data = data[(data['year'] >= year_range[0]) & (data['year'] <= year_range[1])]

        # Show filtered data
        st.write(f"Showing Data from {year_range[0]} to {year_range[1]}")
        st.write(filtered_data)

        # PM2.5 Distribution
        st.write("### PM2.5 Distribution")
        plt.figure(figsize=(10, 6))
        sns.histplot(filtered_data['PM2.5'], kde=True)
        st.pyplot()

        # PM2.5 Time Series Plot
        st.write("### PM2.5 Over Time")
        plt.figure(figsize=(10, 6))
        plt.plot(filtered_data['year'], filtered_data['PM2.5'], color='blue')
        plt.title('PM2.5 Over Time')
        plt.xlabel('year')
        plt.ylabel('PM2.5')
        st.pyplot()

        # Correlation Heatmap
        st.write("### Correlation Heatmap")
        plt.figure(figsize=(10, 6))
        corr = filtered_data[['TEMP', 'DEWP', 'PRES', 'PM2.5']].corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
        st.pyplot()

    elif choice == "Modeling and Prediction":
        st.subheader("Modeling and Prediction")

        # Features and target
        features = ['TEMP', 'DEWP', 'PRES']
        target = 'PM2.5'

        # Train-test split
        X = data[features]
        y = data[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Linear regression model
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Model evaluation
        mse = mean_squared_error(y_test, y_pred)
        st.write(f"### Mean Squared Error: {mse:.4f}")

        # Predictions vs Actual Plot
        st.write("### Predictions vs Actual Values")
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred)
        plt.title('Predictions vs Actual Values')
        plt.xlabel('Actual PM2.5')
        plt.ylabel('Predicted PM2.5')
        st.pyplot()

        # Model Coefficients
        st.write("### Model Coefficients")
        coefficients = pd.DataFrame(model.coef_, features, columns=['Coefficient'])
        st.write(coefficients)

        # Plotting residuals
        st.write("### Residuals Plot")
        residuals = y_test - y_pred
        plt.figure(figsize=(10, 6))
        sns.histplot(residuals, kde=True, color='red')
        plt.title('Residuals Distribution')
        plt.xlabel('Residuals')
        st.pyplot()

if __name__ == "__main__":
    main()
