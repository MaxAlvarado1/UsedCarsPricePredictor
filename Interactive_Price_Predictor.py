import streamlit as st
import joblib 
import pandas as pd
import datetime


UC_Price_Predictor = joblib.load("UC_Price_Predictor.pkl")

full_pipeline = UC_Price_Predictor["full_pipeline"]
column_summary = UC_Price_Predictor["column_summary"]
UCPP_model = UC_Price_Predictor["model"]
data_sample = UC_Price_Predictor["data_sample"]


st.write("""# Used Car Price Predictor

Please visit my github repository to see how this Machine Learning Model was created!

My model uses 23 parameters to predict the price of a used car. Use the left sidebar to input your own parameters to see what my model will predict!
         
The options for the parameters are from my original dataset, but there is no constraints on what combination you choose. So feel free to be unrealistic and have it predict the price for a "2010 Ford Corvette"!

Showing all the data that was used in training my model will be too much to show here, so here is a small sample:
""")


st.write("## Sample Data")

st.write(data_sample)


st.sidebar.header("Input Your Own Parameters!")

def user_input_features(column_summary):
    user_inputs = {} 

    for col, info in column_summary.items():
        if col == "listed_date" and "min" in info and "max" in info: 
            min_date = datetime.datetime.fromtimestamp(info["min"])
            max_date = datetime.datetime.fromtimestamp(info["max"])
            default_date = min_date + (max_date - min_date) / 2  
            selected_date = st.sidebar.date_input(col, default_date)
            user_inputs[col] = selected_date.strftime("%Y-%m-%d") 
            
        elif "min" in info and "max" in info:  
            if info["min"] == 0 and info["max"] == 1:  
                user_inputs[col] = int(st.sidebar.checkbox(col, value=bool(info["min"])))  
            else:
                user_inputs[col] = st.sidebar.slider(
                    col, 
                    min_value=int(info["min"]), 
                    max_value=int(info["max"]), 
                    value=int((info["min"] + info["max"]) / 2)  ,
                    step = 1
                )
        
        elif "unique_values" in info:  
            user_inputs[col] = st.sidebar.selectbox(col, info["unique_values"])
        
        else:
            st.sidebar.write(f"⚠️ Unsupported column type: {col}")

    if "listed_date" in user_inputs:
        user_inputs["listed_date"] = datetime.datetime.strptime(user_inputs["listed_date"], "%Y-%m-%d").timestamp()

    return user_inputs


user_inputs = user_input_features(column_summary)

user_inputs_df = pd.DataFrame([user_inputs])

st.write("## Here is what you inputed:")
user_inputs_show = pd.DataFrame(user_inputs.items(), columns=["Parameter", "Value"])

st.table(user_inputs_show)

user_inputs_df_prepared = full_pipeline.transform(user_inputs_df)

user_inputs_predictions = UCPP_model.predict(user_inputs_df_prepared)

st.write("# Predicted Price: ", user_inputs_predictions[0])



# st.write("## Parameters Used To Predict Price")

# df_columns = pd.DataFrame({"Columns": list(column_summary.keys())})

# Display the table in Streamlit
# st.table(df_columns)