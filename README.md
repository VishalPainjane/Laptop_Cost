# Laptop Price Estimator

I created a model to estimate laptop prices using different types of regression models:

- **Random Forest**
- **Decision Trees**
- **Gradient Boosting**
- **XGBoost**
- **AdaBoost**
- **Linear Regression**

Among these, the **Random Forest Regression** model performed the best, with an accuracy of **85%**.

### Data Source
I gathered the latest data from various online sources to ensure the model is up-to-date with current market specifications. Using the most recent data makes the model more reliable for predicting current laptop prices.

### Outliers and Data Preprocessing
While exploring the data, I found some unusual outliers. I fixed these using the IQR method. However, removing all outliers wasn't feasible because some high-priced laptops are genuine. To handle this, I performed feature engineering to extract more features from existing ones, which helped improve the model's accuracy.

### Model Deployment
This model helps you find the price of a laptop based on its specifications. You can try it out on my Hugging Face account:

Link: [https://huggingface.co/spaces/Vishalpainjane/Laptop_Cost](https://huggingface.co/spaces/Vishalpainjane/Laptop_Cost)

### Running Locally
To run this model on your local machine:

1. **Clone the repository:**
    ```bash
    git clone https://github.com/VishalPainjane/Laptop_Cost.git
    ```

2. **Navigate to the interface folder:**
    ```bash
    cd interface
    ```

3. **Run the interface.py file:**
    ```bash
    python interface.py
    ```

And that's it! The program will run on your local machine.










