import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.figure_factory as ff

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Page setup
st.set_page_config(page_title="📱 Wearable Device Dashboard", layout="wide")
st.title("📊 Wearable Health Devices Dashboard with ML Prediction")

# ✅ Load local CSV file
try:
    df_orig = pd.read_csv('wearable_health_devices_performance_upto_26june2025.csv')

    if df_orig.empty or df_orig.shape[1] == 0:
        st.error("❌ The CSV file is empty or has no columns.")
    else:
        st.success("✅ CSV file loaded successfully!")

        # ------------------------ Preprocessing ------------------------
        df = df_orig.copy()
        df['Test_Date'] = pd.to_datetime(df['Test_Date'], errors='coerce')
        df['Test_Date_day'] = df['Test_Date'].dt.day
        df['Test_Date_month'] = df['Test_Date'].dt.month
        df['Test_Date_year'] = df['Test_Date'].dt.year

        df_processed = df.copy()
        df_processed.drop(columns=['Test_Date', 'Connectivity_Features', 'App_Ecosystem_Support', 'Device_Name'],
                          errors='ignore', inplace=True)

        for col in df_processed.select_dtypes(include='object').columns:
            df_processed = pd.concat([df_processed,
                                      pd.get_dummies(df_processed[col], prefix=col, drop_first=True)], axis=1)
            df_processed.drop(columns=col, inplace=True)

        if 'GPS_Accuracy_Meters' in df_processed.columns:
            imputer = SimpleImputer(strategy='median')
            df_processed['GPS_Accuracy_Meters'] = imputer.fit_transform(df_processed[['GPS_Accuracy_Meters']])

        # ------------------------ Model Training ------------------------
        if 'User_Satisfaction_Rating' in df_processed.columns:
            X = df_processed.drop(columns=['User_Satisfaction_Rating'])
            y = df_processed['User_Satisfaction_Rating']

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            scaler = StandardScaler()
            X_train_sc = scaler.fit_transform(X_train)
            X_test_sc = scaler.transform(X_test)

            model = RandomForestRegressor(max_samples=0.75, random_state=42)
            model.fit(X_train_sc, y_train)
            y_pred = model.predict(X_test_sc)

            r2 = r2_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)

            # Feature Importance
            importances = model.feature_importances_
            feature_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
            feature_df = feature_df.sort_values(by='Importance', ascending=False)
            top_features = feature_df.head(5)['Feature'].tolist()

            # ------------------------ Tabs ------------------------
            tab1, tab2, tab3, tab4 = st.tabs([
                "📈 Model Performance",
                "📊 Visual Analytics",
                "📉 Correlation Heatmap",
                "🎯 Predict Satisfaction"
            ])

            # 📈 Model Performance Tab
            with tab1:
                st.subheader("📈 Model Performance Metrics")
                col1, col2, col3 = st.columns(3)
                col1.metric("R² Score", f"{r2:.4f}")
                col2.metric("MSE", f"{mse:.2f}")
                col3.metric("MAE", f"{mae:.2f}")

                st.subheader("🔍 Top 10 Feature Importances")
                fig_imp = px.bar(feature_df.head(10), x='Importance', y='Feature', orientation='h',
                                 title="Feature Importance", color='Importance', color_continuous_scale='Viridis')
                st.plotly_chart(fig_imp, use_container_width=True)

            # 📊 Visual Analytics Tab
            with tab2:
                if 'Brand' in df_orig.columns:
                    st.subheader("📈 Average User Satisfaction by Brand")
                    avg_rating = df_orig.groupby("Brand")["User_Satisfaction_Rating"].mean().reset_index()
                    fig1 = px.bar(avg_rating, x="Brand", y="User_Satisfaction_Rating",
                                  color="User_Satisfaction_Rating", title="Average Satisfaction by Brand")
                    st.plotly_chart(fig1, use_container_width=True)

                st.subheader("💰 Price vs. User Satisfaction")
                if 'Price_USD' in df_orig.columns and 'Model' in df_orig.columns:
                    fig2 = px.scatter(df_orig, x="Price_USD", y="User_Satisfaction_Rating", color="Category",
                                      size="Battery_Life_Hours", hover_name="Model",
                                      title="Price vs Satisfaction by Category")
                    st.plotly_chart(fig2, use_container_width=True)

                st.subheader("🔋 Battery Life vs Performance Score")
                filtered_df = df_orig[df_orig['Brand'] != 'Garmin']
                fig3 = px.line(filtered_df.sort_values("Battery_Life_Hours"),
                               x="Battery_Life_Hours",
                               y="Performance_Score",
                               color="Brand",
                               title="Battery Life vs Performance Score (excluding Garmin)")
                st.plotly_chart(fig3, use_container_width=True)

                st.subheader("📡 Accuracy Comparison Across Models")
                fig4 = go.Figure()
                acc_data = df_orig[['Model', 'Heart_Rate_Accuracy_Percent',
                                    'Step_Count_Accuracy_Percent', 'Sleep_Tracking_Accuracy_Percent']].dropna()
                for i in range(len(acc_data)):
                    fig4.add_trace(go.Scatterpolar(
                        r=acc_data.iloc[i, 1:].values,
                        theta=['Heart Rate', 'Step Count', 'Sleep Tracking'],
                        fill='toself',
                        name=acc_data.iloc[i]['Model']
                    ))
                fig4.update_layout(polar=dict(radialaxis=dict(visible=True, range=[70, 100])), showlegend=True)
                st.plotly_chart(fig4, use_container_width=True)

                st.subheader("📦 Performance Score by Category")
                fig5 = px.box(df_orig, x="Category", y="Performance_Score", color="Category")
                st.plotly_chart(fig5, use_container_width=True)

                st.subheader("🧁 Device Category Distribution")
                fig6 = px.pie(df_orig, names="Category", hole=0.4)
                st.plotly_chart(fig6, use_container_width=True)

            # 📉 Correlation Heatmap Tab
            with tab3:
                st.subheader("📉 Correlation Heatmap")
                corr = df_processed.select_dtypes(include=np.number).corr().round(2)
                fig_heatmap = ff.create_annotated_heatmap(
                    z=corr.values,
                    x=list(corr.columns),
                    y=list(corr.index),
                    colorscale="RdBu",
                    showscale=True,
                    annotation_text=corr.values.astype(str)
                )
                fig_heatmap.update_layout(width=1000, height=600, margin=dict(l=50, r=50, t=50, b=50))
                st.plotly_chart(fig_heatmap, use_container_width=True)

            # 🎯 Prediction Tab
            with tab4:
                st.subheader("🎯 Predict User Satisfaction")
                input_data = []
                for feature in top_features:
                    val = st.number_input(f"{feature}", value=float(df_processed[feature].mean()))
                    input_data.append(val)

                if st.button("Predict"):
                    user_df = pd.DataFrame([X.mean().to_dict()])
                    for f, v in zip(top_features, input_data):
                        user_df[f] = v
                    user_df_scaled = scaler.transform(user_df)
                    prediction = model.predict(user_df_scaled)[0]
                    st.success(f"✅ Predicted User Satisfaction: {prediction:.2f}")

except pd.errors.EmptyDataError:
    st.error("❌ The CSV file is empty. Please check the file.")
except FileNotFoundError:
    st.error("❌ File not found. Make sure 'wearable_health_devices_performance_upto_26june2025.csv' exists.")
except Exception as e:
    st.error(f"⚠️ An unexpected error occurred: {e}")
