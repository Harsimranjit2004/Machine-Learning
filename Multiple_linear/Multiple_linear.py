# import streamlit as st
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error, r2_score
# from sklearn.preprocessing import LabelEncoder
# import base64

# def load_data(file):
#     data = pd.read_csv(file)
#     return data

# def preprocess_data(df, selected_columns, target_variable):
#     X = df[selected_columns]
#     y = df[target_variable]
    
#     # Encode categorical variables
#     cat_cols = X.select_dtypes(include=['object']).columns.tolist()
#     if cat_cols:
#         for col in cat_cols:
#             encoder = LabelEncoder()
#             X[col] = encoder.fit_transform(X[col])

#     return X, y

# def build_model(X_train, y_train):
#     model = LinearRegression()
#     model.fit(X_train, y_train)
#     return model

# def evaluate_model(model, X_test, y_test):
#     y_pred = model.predict(X_test)
#     mse = mean_squared_error(y_test, y_pred)
#     r2 = r2_score(y_test, y_pred)
#     return mse, r2, y_pred

# def plot_barplot_categorical(df, input_column, target_column):
#     mean_output = df.groupby(input_column)[target_column].mean()
#     plt.figure(figsize=(8, 6))
#     sns.barplot(x=mean_output.index, y=mean_output.values)
#     plt.xlabel(input_column)
#     plt.ylabel('Mean ' + target_column)
#     plt.title(f'Mean {target_column} by {input_column}')
#     st.pyplot()

# def plot_scatterplot_categorical(df, input_column, target_column):
#     plt.figure(figsize=(8, 6))
#     sns.stripplot(x=input_column, y=target_column, data=df, jitter=True, alpha=0.5)
#     plt.xlabel(input_column)
#     plt.ylabel(target_column)
#     plt.title(f'{input_column} vs. {target_column}')
#     st.pyplot()
    
# def plot_correlation_heatmap(df):
#     plt.figure(figsize=(10, 8))
#     sns.heatmap(df.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
#     plt.title('Correlation Heatmap')
#     st.pyplot()

# def main():
#     st.set_page_config(layout="wide")
#     st.title("Multiple Linear Regression Analysis")

#     st.write("""
#     Welcome to the Multiple Linear Regression Analysis web application.
#     """)

#     uploaded_file = st.file_uploader("Upload your input CSV file", type=["csv"])

#     if uploaded_file is not None:
#         data = load_data(uploaded_file)
#         st.write("Uploaded file preview:")
#         st.write(data.head())

#         st.header('Statistical Summary')
#         st.write(data.describe())

#         selected_columns = st.multiselect('Select independent variables', data.columns)
#         target_variable = st.selectbox('Select dependent variable', data.columns)

#         if st.button('Run Analysis'):
#             X, y = preprocess_data(data, selected_columns, target_variable)

#             if len(selected_columns) > 0 and target_variable:
#                 X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#                 model = build_model(X_train, y_train)
#                 mse, r2, y_pred = evaluate_model(model, X_test, y_test)

#                 st.subheader('Model Performance')
#                 st.write('Mean Squared Error:', mse)
#                 st.write('R-squared:', r2)

#                 st.subheader('Intercept and Coefficients')
#                 st.write('Intercept:', model.intercept_)
#                 st.write('Coefficients:')
#                 for i, col in enumerate(selected_columns):
#                     st.write(f'{col}: {model.coef_[i]}')

#                 st.subheader('Feature Importance')
#                 feature_importance = pd.DataFrame({'Feature': selected_columns, 'Importance': model.coef_})
#                 st.write(feature_importance)

#                 st.subheader('Visualizations')

#                 st.write('1. Correlation Heatmap')
#                 plot_correlation_heatmap(data)

#                 st.write('2. Bar Plot for Categorical Columns')
#                 categorical_columns = [col for col in selected_columns if data[col].dtype == 'object']
#                 if categorical_columns:
#                     for col in categorical_columns:
#                         plot_barplot_categorical(data, col, target_variable)
#                 else:
#                     st.write("No categorical columns selected.")

#                 st.write('3. Scatter Plot for Numerical Columns')
#                 numerical_columns = [col for col in selected_columns if data[col].dtype != 'object']
#                 if numerical_columns:
#                     for col in numerical_columns:
#                         plot_scatterplot_categorical(data, col, target_variable)
#                 else:
#                     st.write("No numerical columns selected.")

#                 # Download predicted values
#                 with st.expander("Download Predicted Values"):
#                     df_pred = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
#                     csv = df_pred.to_csv(index=False)
#                     b64 = base64.b64encode(csv.encode()).decode()
#                     href = f'<a href="data:file/csv;base64,{b64}" download="predicted_values.csv">Download CSV</a>'
#                     st.markdown(href, unsafe_allow_html=True)

#             else:
#                 st.warning('Please select independent and dependent variables.')
#     else:
#         st.warning('Please upload a CSV file.')



# def IntroPage():
#     st.markdown('''
#     <h1 style = "text-align:center" color: "#eb6b40">Multiple Lineaer Regression</h1>''',unsafe_allow_html=True)
#     st.write('''
#     ***Multiple Linear Regression*** 
# Multiple linear regression is a statistical method used to analyze the relationship between two or more independent variables  and a dependent variable . It extends simple linear regression, which examines the relationship between one independent variable and a dependent variable, to situations where multiple independent variables may influence the dependent variable.''')

# #st.set_page_config(layout='centered')

# if __name__ == "__main__":
# import streamlit as st
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error, r2_score
# from sklearn.impute import SimpleImputer
# import matplotlib.pyplot as plt
# import seaborn as sns
# import io

# def handle_missing_values(df):
#     # Impute missing values with mean
#     imputer = SimpleImputer(strategy='mean')
#     numerical_cols = df.select_dtypes(include=np.number).columns
#     df[numerical_cols] = imputer.fit_transform(df[numerical_cols])
#     return df

# def MultipleLinearRegressionDemo():
#     st.title("Multiple Linear Regression Demo")

#     # Upload dataset
#     uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
#     if uploaded_file is not None:
#         st.write("Uploaded file preview:")
#         df = pd.read_csv(uploaded_file)

#         # Handle missing values
#         df = handle_missing_values(df)

#         # Show dataset statistics
#         st.subheader("Dataset Statistics:")
#         st.write(df.describe())

#         # Show a preview of the dataset
#         st.subheader("Data Preview:")
#         st.write(df.head())

#         # Choose input and output columns
#         numerical_cols = df.select_dtypes(include=np.number).columns
#         input_columns = st.multiselect("Select input columns (numerical)", numerical_cols)
#         output_column = st.selectbox("Select the output column (numerical)", numerical_cols)

#         if not input_columns or not output_column:
#             st.warning("Please select input and output columns.")
#             return

#         X = df[input_columns]
#         y = df[output_column]

#         # Train-test split
#         test_size = st.slider("Test Size", min_value=0.1, max_value=0.5, step=0.05, value=0.2)
#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

#         # Multiple Linear Regression model
#         model = LinearRegression()
#         model.fit(X_train, y_train)

#         # Predictions on test set
#         y_pred = model.predict(X_test)

#         # Evaluate model
#         mse = mean_squared_error(y_test, y_pred)
#         r2 = r2_score(y_test, y_pred)

#         # Display coefficients and intercept
#         st.subheader("Model Coefficients:")
#         coeff_df = pd.DataFrame({
#             'Feature': input_columns,
#             'Coefficient': model.coef_
#         })
#         st.write(coeff_df)

#         st.subheader("Intercept:")
#         st.write(model.intercept_)

#         st.subheader("Model Performance:")
#         st.write(f"Mean Squared Error: {mse}")
#         st.write(f"R-Squared Score (R2): {r2}")

#         # Visualization - Scatter plot
#         st.subheader("Scatter Plot - Actual vs. Predicted")
#         fig, ax = plt.subplots()
#         ax.scatter(y_test, y_pred, color='blue')
#         ax.plot(y_test, y_test, color='red', linewidth=2)
#         ax.set_xlabel("Actual")
#         ax.set_ylabel("Predicted")
#         st.pyplot(fig)

#         # Visualization - Distribution plot
#         st.subheader("Residuals Distribution")
#         residuals = y_test - y_pred
#         sns.histplot(residuals, kde=True)
#         st.pyplot()

#         # Relationship between input columns and output
#         st.subheader("Relationship between Input and Output:")
#         for col in input_columns:
#             fig, ax = plt.subplots()
#             ax.scatter(X_test[col], y_test, color='blue', label='Actual')
#             ax.plot(X_test[col], model.predict(X_test), color='red', linewidth=2, label='Predicted')
#             ax.set_xlabel(col)
#             ax.set_ylabel(output_column)
#             ax.legend()
#             st.pyplot(fig)

#         # Download model
#         st.subheader("Download Trained Model")
#         model_bytes = io.BytesIO()
#         pd.to_pickle(model, model_bytes)
#         st.download_button(label="Download Trained Model", data=model_bytes, file_name="multiple_linear_regression_model.pkl", key="download_model")

# # Run the app
# if __name__ == '__main__':
#     MultipleLinearRegressionDemo()
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import io
import plotly.graph_objects as go
import plotly.express as px
from sklearn.datasets import make_regression

def handle_missing_values(df):
    # Impute missing values with mean
    imputer = SimpleImputer(strategy='mean')
    numerical_cols = df.select_dtypes(include=np.number).columns
    df[numerical_cols] = imputer.fit_transform(df[numerical_cols])
    return df

def upload_data():
    st.title("Multiple Linear Regression Demo")
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file is not None:
        st.write("Uploaded file preview:")
        df = pd.read_csv(uploaded_file)
        return df

def show_dataset_statistics(df):
    st.subheader("Dataset Statistics:")
    st.write(df.describe())

def show_data_preview(df):
    st.subheader("Data Preview:")
    st.write(df.head())

def select_columns(df):
    numerical_cols = df.select_dtypes(include=np.number).columns
    input_columns = st.multiselect("Select input columns (numerical)", numerical_cols)
    output_column = st.selectbox("Select the output column (numerical)", numerical_cols)
    return input_columns, output_column

def train_test_split_data(X, y):
    test_size = st.slider("Test Size", min_value=0.1, max_value=0.5, step=0.05, value=0.2)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    return X_train, X_test, y_train, y_test

def train_linear_regression(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return y_pred, mse, r2

def scatter_plot(input_columns, output_column, X_test, y_test, y_pred):
    for col in input_columns:
        fig, ax = plt.subplots()
        sns.scatterplot(x=X_test[col], y=y_test, ax=ax, label='Actual')
        sns.lineplot(x=X_test[col], y=y_pred, color='red', ax=ax, label='Predicted')
        ax.set_xlabel(col)
        ax.set_ylabel(output_column)
        ax.legend()
        st.pyplot(fig)

def residuals_plot(y_test, y_pred):
    fig, ax = plt.subplots()
    residuals = y_test - y_pred
    sns.scatterplot(x=y_pred, y=residuals, ax=ax)
    ax.axhline(y=0, color='red', linestyle='--')
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Residuals")
    st.pyplot(fig)

def feature_importance_plot(input_columns, model):
    coeff_df = pd.DataFrame({
        'Feature': input_columns,
        'Coefficient': model.coef_
    })
    coeff_df['abs_coefficient'] = np.abs(coeff_df['Coefficient'])
    coeff_df = coeff_df.sort_values('abs_coefficient', ascending=False)

    fig, ax = plt.subplots()
    sns.barplot(x='Coefficient', y='Feature', data=coeff_df, palette='viridis')
    ax.set_xlabel("Coefficient")
    ax.set_ylabel("Feature")
    ax.set_title("Feature Importance")
    st.pyplot(fig)

def histograms(df, numerical_cols):
    st.subheader("Histograms for Numerical Columns:")
    for col in numerical_cols:
        fig, ax = plt.subplots()
        sns.histplot(df[col], kde=True, ax=ax)
        ax.set_xlabel(col)
        ax.set_ylabel("Count")
        st.pyplot(fig)

def correlation_heatmap(df):
    st.subheader("Correlation Matrix Heatmap:")
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    st.pyplot()

def pairplot(input_columns, output_column, df):
    st.subheader("Pairplot (Limited):")
    cols = input_columns + [output_column]
    sns.pairplot(df[cols].sample(n=1000, random_state=42), diag_kind='kde')
    st.pyplot()

def download_trained_model(model):
    st.subheader("Download Trained Model")
    model_bytes = io.BytesIO()
    pd.to_pickle(model, model_bytes)
    st.download_button(label="Download Trained Model", data=model_bytes, file_name="multiple_linear_regression_model.pkl", key="download_model")

def MultipleLinearRegressionDemo():
    df = upload_data()

    if df is not None:
        # Handle missing values
        df = handle_missing_values(df)

        show_dataset_statistics(df)
        show_data_preview(df)

        # Choose input and output columns
        input_columns, output_column = select_columns(df)

        if not input_columns or not output_column:
            st.warning("Please select input and output columns.")
            return

        X = df[input_columns]
        y = df[output_column]

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split_data(X, y)

        # Train Linear Regression model
        model = train_linear_regression(X_train, y_train)

        # Evaluate model
        y_pred, mse, r2 = evaluate_model(model, X_test, y_test)

        # Display coefficients and intercept
        st.subheader("Model Coefficients:")
        coeff_df = pd.DataFrame({
            'Feature': input_columns,
            'Coefficient': model.coef_
        })
        st.write(coeff_df)

        st.subheader("Intercept:")
        st.write(model.intercept_)

        st.subheader("Model Performance:")
        st.write(f"Mean Squared Error: {mse}")
        st.write(f"R-Squared Score (R2): {r2}")

        # Visualization options
        st.subheader("Visualizations:")
        if st.button("Scatter Plot with Predictions"):
            scatter_plot(input_columns, output_column, X_test, y_test, y_pred)

        if st.button("Residuals Plot"):
            residuals_plot(y_test, y_pred)

        if st.button("Feature Importance"):
            feature_importance_plot(input_columns, model)

        if st.button("Histograms"):
            histograms(df, input_columns)

        if st.button("Correlation Heatmap"):
            correlation_heatmap(df)

        if st.button("Pairplot"):
            pairplot(input_columns, output_column, df)

        # Download model
        download_trained_model(model)


def main():
    st.markdown('''
<h1 style = "text-align:center"  "color: #eb6b40">Multiple Linear Regression</h1>''', unsafe_allow_html=True)
    st.markdown('''
***Multiple Linear Regression*** is a statistical method that extends the principles of simple linear regression to model the relationship between a dependent variable and two or more independent variables.
''')
    st.markdown('''In simple language multiple linear regression tries to find best-fit hyperplane on the data. Like for ''')
    st.markdown('''2d->Line''')
    st.markdown('''3d->Plane''')
    st.markdown('''nd->hyperplanes''')
    X,y  = make_regression(n_samples = 100, n_features=2, n_informative=2, n_targets=1, noise=50)
    df = pd.DataFrame({'feature1':X[:,0],'feature2':X[:,1], 'target':y})

    # fig = px.scatter_3d(df,x= 'feature1', y = 'feature2', z = 'target')
    # st.plotly_chart(fig)




    # fig = px.scatter_3d(df, x='feature1', y='feature2', z='target')
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

# Fit the multiple linear regression model
    model = LinearRegression()
    model.fit(X_scaled, y)

# Make predictions
    y_pred = model.predict(X_scaled)
# Add the    decision plane
    fig = px.scatter_3d(df, x='feature1', y='feature2', z='target')

# Add the decision plane
    x_plane = np.linspace(df['feature1'].min(), df['feature1'].max(), 10)
    y_plane = np.linspace(df['feature2'].min(), df['feature2'].max(), 10)
    X_plane, Y_plane = np.meshgrid(x_plane, y_plane)
    Z_plane = model.coef_[0] * X_plane + model.coef_[1] * Y_plane + model.intercept_
    # fig = plt.figure()
    # ax = fig.add_subplot(111,projection='3d')
    # ax.plot_surface(X_plane,Y_plane,Z_plane)
    fig.add_trace(go.Surface(z=Z_plane, x=X_plane, y=Y_plane, colorscale='Viridis'))

    # fig = go.Figure(data = [go.surface(z = Z_plane, x = X_plane, y = Y_plane)])
    st.plotly_chart(fig)


    st.markdown('''
    In the above Graph there is plane that is tries to pass the given points 
''')

    
    
    
# Run the app
if __name__ == '__main__':
    main()
