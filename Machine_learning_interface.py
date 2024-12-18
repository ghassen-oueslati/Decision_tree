import streamlit as st
import pandas as pd
import joblib

# Load the trained model and label encoders
@st.cache_data
def load_model_and_encoders():
    model = joblib.load(r"C:\Users\ouesl\OneDrive\Desktop\Esprit\3ème\ML\decision_tree_model.pkl")
    encoders = joblib.load(r"C:\Users\ouesl\OneDrive\Desktop\Esprit\3ème\ML\label_encoders.pkl")
    return model, encoders

# Function to preprocess user input
def preprocess_input(data, label_encoders):
    for column in data.columns:
        if column in label_encoders:
            le = label_encoders[column]
            try:
                data[column] = le.transform(data[column])
            except ValueError:
                st.error(f"Invalid value in '{column}'. Allowed values: {list(le.classes_)}")
                return None
    return data

# Streamlit app layout
def main():
    st.title("Clothing Recommendation Prediction")
    st.write("This app predicts whether a clothing item will be **recommended** or **not recommended** based on its features.")

    # Load model and encoders
    model, label_encoders = load_model_and_encoders()

    # Sidebar inputs for user data
    st.sidebar.header("Input Features")
    style = st.sidebar.selectbox("Style", ["Casual", "Sexy", "vintage", "cute", "Brief"])
    price = st.sidebar.selectbox("Price", ["Low", "Medium", "High"])
    rating = st.sidebar.slider("Rating", 0.0, 5.0, step=0.1, value=4.0)
    size = st.sidebar.selectbox("Size", ["S", "M", "L", "XL", "free"])
    season = st.sidebar.selectbox("Season", ["Summer", "Spring", "Autumn", "Winter"])
    neckline = st.sidebar.selectbox("NeckLine", ["round-neck", "v-neck", "boat-neck"])
    sleeve_length = st.sidebar.selectbox("SleeveLength", ["sleeveless", "short", "full", "half-sleeve"])
    material = st.sidebar.selectbox("Material", ["cotton", "silk", "polyester", "microfiber", "Unknown"])
    fabric_type = st.sidebar.selectbox("FabricType", ["chiffon", "broadcloth", "corduroy", "Unknown"])
    pattern_type = st.sidebar.selectbox("Pattern Type", ["solid", "print", "striped", "animal", "dot"])

    # Create input data
    input_data = pd.DataFrame({
        'Style': [style],
        'Price': [price],
        'Rating': [rating],
        'Size': [size],
        'Season': [season],
        'NeckLine': [neckline],
        'SleeveLength': [sleeve_length],
        'Material': [material],
        'FabricType': [fabric_type],
        'Pattern Type': [pattern_type]
    })

    st.write("### User Input:")
    st.table(input_data)

    # Preprocess input and predict
    if st.button("Predict Recommendation"):
        processed_data = preprocess_input(input_data.copy(), label_encoders)
        if processed_data is not None:
            prediction = model.predict(processed_data)
            if prediction[0] == 1:
                st.success("🎉 **Recommended!**")
            else:
                st.error("❌ **Not Recommended!**")

if __name__ == "__main__":
    main()