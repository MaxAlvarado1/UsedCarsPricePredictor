import streamlit as st
import gdown
import joblib 
import traceback
import pandas as pd
import datetime
import os

@st.cache_resource
def load_model():
    url = 'https://drive.google.com/uc?export=download&id=1hM4TSC2wFiphs_8dAMLf3ZYvSoct633z'
    output = 'model.pkl'

    st.write("📦 Downloading model...")
    gdown.download(url, output, quiet=False)
    
    # Check if file exists after download
    if os.path.exists(output):
        st.write(f"📥 Download complete, file size: {os.path.getsize(output)} bytes")
    else:
        st.error("❌ Model download failed!")
        return None

    try:
        st.write("📥 Loading model...")
        model = joblib.load(output) 
        st.success("✅ Model loaded successfully")
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.text(traceback.format_exc())
        return None

full_pipeline = joblib.load('pipeline.pkl')
column_summary = joblib.load('summary.pkl')
data_sample = joblib.load('sample.pkl')


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

st.write("# Used Car Price Predictor")

st.sidebar.header("Customize the Parameters")
st.sidebar.markdown("Adjust the settings below, then click **Predict** at the bottom to see the estimated price.")

user_inputs = user_input_features(column_summary)

if st.sidebar.button("Predict"):
    UCPP_model = load_model()
    if UCPP_model is None:
        st.stop()
    user_inputs_df = pd.DataFrame([user_inputs])

    user_inputs_df_prepared = full_pipeline.transform(user_inputs_df)
    user_inputs_predictions = UCPP_model.predict(user_inputs_df_prepared)

    predicted_price = user_inputs_predictions[0]
    formatted_price = f"${predicted_price:,.0f}"
    st.markdown(f"# Predicted Price: <span style='color:green'>{formatted_price}</span>", unsafe_allow_html=True)

    st.write("## Here is what you inputed:")
    user_inputs_show = pd.DataFrame(
        [(k, str(v)) for k, v in user_inputs.items()],
        columns=["Parameter", "Value"]
    )
    st.table(user_inputs_show)

st.write("")

st.markdown("""  
Please visit my [GitHub repository](https://github.com/MaxAlvarado1/UsedCarsPricePredictor) to see how this Machine Learning Model was created!

My model uses 23 parameters to predict the price of a used car. Use the left sidebar to input your own parameters to see what my model will predict!
         
The options for the parameters are from my original dataset, but there is no constraint on what combination you choose. So feel free to be unrealistic and have it predict the price for a "2010 Ford Corvette"!

Showing all the data that was used in training my model will be too much to show here, so here is a small sample:
""")

st.write("## Sample Data")

st.write(data_sample)