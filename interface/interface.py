import gradio as gr
import dill as pickle

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, mean_squared_error
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from joblib import load

data = pickle.load(open('data/data3.pkl', 'rb'))


def cal_gradio(status, brand, cpu_brand, cpu, ram, storage, storage_type, gpu, screen, touch):
    
    df = pd.DataFrame({
        'Status': [status],
        'Brand': [brand],
        'CPU': [cpu],
        'RAM': [ram],
        'Storage': [storage],
        'Storage type': [storage_type],
        'GPU': [gpu],
        'Screen': [screen],
        'Touch': [touch],
        'CPU_brand': [cpu_brand]
    })
    
    global data
    
    data = pd.concat([data, df], ignore_index=True)
    
    categorical_columns = data.select_dtypes(include=['object']).columns
    element_to_number = {}
    for col in categorical_columns:
        unique_elements = data[col].unique()
        for idx, elem in enumerate(unique_elements):
            element_to_number[(col, elem)] = idx
    
    for col in categorical_columns:
        data[col] = data[col].apply(lambda x: element_to_number[(col, x)])
    
    last_row = data.iloc[-1]
    df = pd.DataFrame([last_row], columns=data.columns)
    data = data.iloc[:-1]
    
    if 'Final Price' in df.columns:
        df = df.drop(columns=['Final Price']) 
        
    if 'Final Price' in data.columns:
        X = data.drop(columns=['Final Price'])
        y = np.log(data['Final Price'])
        
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=1)
    
    rf = RandomForestRegressor(n_estimators=300, random_state=6, max_samples=0.6, max_features=0.08, max_depth=21)
    rf.fit(X_train, y_train)
    
    y_pred = rf.predict(df)
    
    price = np.exp(float(y_pred)) * 91.39

    price = round(price,-3)

    price-=1
    return price


status_dropdown = gr.Dropdown(choices=['New', 'Refurbished'], label="Condition", info="Select whether the laptop is new or refurbished.")
brand_dropdown = gr.Dropdown(choices=['Apple', 'Razer', 'Asus', 'HP', 'Alurin', 'MSI', 'Lenovo', 'Medion', 'Acer', 'Gigabyte', 'Dell', 'LG', 'Microsoft'], label="Preferred Brand", info="Choose the brand of your desired laptop.")
cpu_brand_dropdown = gr.Dropdown(choices=['Intel', 'AMD', 'Apple'], label="Choose Your CPU Brand", info="Select the brand of the CPU.")
cpu_dropdown = gr.Dropdown(choices=['Intel Core i9', 'Intel Core i7', 'Intel Core i5', 'Intel Core i3', 'Intel Evo Core i7', 'Intel Evo Core i5', 'Intel Celeron',
                                    'Apple M1', 'Apple M1 Pro', 'Apple M2', 'Apple M2 Pro', 'AMD Ryzen 9', 'AMD Ryzen 7', 'AMD Ryzen 5', 'AMD Ryzen 3',
                                    'AMD Radeon 9', 'AMD Radeon 5', 'AMD Athlon', 'AMD 3020e', 'Intel Pentium', 'AMD 3015e', 'Intel Core M3', 'AMD 3015Ce'], label="Select Your CPU", info="Choose the CPU model.")
ram_dropdown = gr.Dropdown(choices=[64, 32, 16, 12, 8, 4], label="Memory (RAM) in GB", info="Select the amount of RAM.")
storage_dropdown = gr.Dropdown(choices=[4000, 3000, 2000, 1000, 512, 500, 256, 240, 128, 64, 32, 0], label="Storage Capacity in GB", info="Choose the storage capacity.")
storage_type_dropdown = gr.Dropdown(choices=['SSD', 'eMMC', 'NO_STORAGE'], label="Type of Storage", info="Select the type of storage.")
gpu_dropdown = gr.Dropdown(choices=['NO_GPU', 'RTX 4090', 'RTX 4080', 'RTX 4060', 'RTX 4070', 'RTX 4050', 'RTX 3080', 'RTX 3070', 'RTX 3060', 'RTX 3050', 'RTX 2080', 'RTX 2070', 'RTX 2060', 'RTX 2050', 'RTX A1000', 'RTX 3000', 'RTX A5500', 'RTX A3000', 'RTX A2000',
                                    'GTX 1660', 'GTX 1650', 'GTX 1050', 'GTX 1070', 'RX 6500M', 'RX 7600S', 'RX 6800S', 'RX 6700M', 'MX 550', 'MX 330', 'MX 450', 'MX 130',
                                    'A 370M', 'A 730M', 'T 1200', 'T 2000', 'T 500', 'T 550', 'T 600', 'T 1000', '610 M', 'Radeon Pro 5500M', 'Radeon RX 6600M', 'Radeon Pro RX 560X', 'Radeon Pro 5300M', 'P 500'], label="Choose Your GPU", info="Select the GPU model.")
screen_slider = gr.Slider(label="Screen Size (in Inches)", minimum=10, maximum=17, step=1, info="Select the screen size in inches.")
touch_checkbox = gr.Checkbox(label="Includes Touch Screen", info="Check if the laptop includes a touch screen.")

iface = gr.Interface(
    fn=cal_gradio,
    inputs=[status_dropdown, brand_dropdown, cpu_brand_dropdown, cpu_dropdown, ram_dropdown,
            storage_type_dropdown,storage_dropdown,  gpu_dropdown, screen_slider,
            touch_checkbox],
    outputs="number",
    title="Laptop Cost Estimator",
    description="<div style='text-align: center;'>Use this model to estimate the cost of a laptop based on your selected specifications. Simply choose your desired options from the dropdown menus and see the estimated price.</div>",
    )

iface.launch()