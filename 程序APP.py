import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# 加载保存的随机森林模型
model = joblib.load('XGBoost.pkl')

# 特征范围定义（根据提供的特征范围和数据类型）
feature_ranges = {
    "age": {"type": "numerical", "min": 18, "max": 100, "default": 71},
    "BAR": {"type": "numerical", "min": 0, "max": 100, "default": 40},
    "FOIS": {"type": "numerical", "min": 1, "max": 7, "default": 3},
    "Thalamic_injury": {"type": "categorical", "options": [0, 1], "default": 0},
    "Cerebellar_injury":{"type": "categorical", "options": [0, 1], "default": 0},
    "Tracheotomy": {"type": "categorical", "options": [0, 1], "default": 0},
    "Labial_motor_function":{"type": "categorical", "options": [0, 1], "default": 0},
    "Lingual_motor_function":{"type": "categorical", "options": [0, 1], "default": 0},
    "Swallowing_reflex": {"type": "categorical", "options": [0, 1], "default": 0}
}

# Streamlit 界面
st.title("Prediction Model with SHAP Visualization")

# 动态生成输入项
st.header("Enter the following feature values:")
feature_values = []
for feature, properties in feature_ranges.items():
    if properties["type"] == "numerical":
        value = st.number_input(
            label=f"{feature} ({properties['min']} - {properties['max']})",
            min_value=float(properties["min"]),
            max_value=float(properties["max"]),
            value=float(properties["default"]),
        )
    elif properties["type"] == "categorical":
        value = st.selectbox(
            label=f"{feature} (Select a value)",
            options=properties["options"],
        )
    feature_values.append(value)

# 转换为模型输入格式
features = np.array([feature_values])

if st.button("Make Prediction"):  # 如果点击了预测按钮
    # Predict the class and probabilities
    predicted_class = model.predict(features)[0]  # 预测心脏病类别
    predicted_proba = model.predict_proba(features)[0]  # 预测各类别的概率

    # Display the prediction results
    st.write(f"**Predicted Class:** {predicted_class}")  # 显示预测的类别
    st.write(f"**Prediction Probabilities:** {predicted_proba}")  # 显示各类别的预测概率

    # Generate advice based on the prediction result
    probability = predicted_proba[predicted_class] * 100  # 根据预测类别获取对应的概率，并转化为百分比

    if predicted_class == 1:  # 如果预测为心脏病
        advice = (
            f"According to our model, your risk of heart disease is high. "
            f"The probability of you having heart disease is {probability:.1f}%. "
            "Although this is just a probability estimate, it suggests that you might have a higher risk of heart disease. "
            "I recommend that you contact a cardiologist for further examination and assessment, "
            "to ensure you receive an accurate diagnosis and necessary treatment."
        )  # 如果预测为心脏病，给出相关建议
    else:  # 如果预测为无心脏病
        advice = (
            f"According to our model, your risk of heart disease is low. "
            f"The probability of you not having heart disease is {probability:.1f}%. "
            "Nevertheless, maintaining a healthy lifestyle is still very important. "
            "I suggest that you have regular health check-ups to monitor your heart health, "
            "and seek medical attention if you experience any discomfort."
        )  # 如果预测为无心脏病，给出相关建议

    st.write(advice)  # 显示建议

    # Visualize the prediction probabilities
    sample_prob = {
        'Class_0': predicted_proba[0],  # 类别0的概率
        'Class_1': predicted_proba[1]  # 类别1的概率
    }

    # Set figure size
    plt.figure(figsize=(10, 3))  # 设置图形大小

    # Create bar chart
    bars = plt.barh(['Not Sick', 'Sick'], 
                    [sample_prob['Class_0'], sample_prob['Class_1']], 
                    color=['#512b58', '#fe346e'])  # 绘制水平条形图

    # Add title and labels, set font bold and increase font size
    plt.title("Prediction Probability for Patient", fontsize=20, fontweight='bold')  # 添加图表标题，并设置字体大小和加粗
    plt.xlabel("Probability", fontsize=14, fontweight='bold')  # 添加X轴标签，并设置字体大小和加粗
    plt.ylabel("Classes", fontsize=14, fontweight='bold')  # 添加Y轴标签，并设置字体大小和加粗

    # Add probability text labels, adjust position to avoid overlap, set font bold
    for i, v in enumerate([sample_prob['Class_0'], sample_prob['Class_1']]):  # 为每个条形图添加概率文本标签
        plt.text(v + 0.0001, i, f"{v:.2f}", va='center', fontsize=14, color='black', fontweight='bold')  # 设置标签位置、字体加粗

    # Hide other axes (top, right, bottom)
    plt.gca().spines['top'].set_visible(False)  # 隐藏顶部边框
    plt.gca().spines['right'].set_visible(False)  # 隐藏右边框

    # Show the plot
    st.pyplot(plt)  # 显示图表

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(pd.DataFrame([feature_values], columns=feature_ranges.keys()))

    # 生成 SHAP 力图
    class_index = predicted_class  # 当前预测类别
    shap_fig = shap.force_plot(
        explainer.expected_value[class_index],
        shap_values[:,:,class_index],
        pd.DataFrame([feature_values], columns=feature_ranges.keys()),
        matplotlib=True,
    )
    # 保存并显示 SHAP 图
    plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)
    st.image("shap_force_plot.png")