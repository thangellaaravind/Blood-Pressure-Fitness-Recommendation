import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Data for recommendations
data = {
    "Recommendation": [
        "Reduce salt intake.", "Practice yoga and meditation daily.", "Walk at least 30 minutes daily.",
        "Focus on high-fiber foods.", "Limit caffeine and alcohol.", "Engage in strength training twice a week.",
        "Drink plenty of water throughout the day.", "Include omega-3 rich foods like fish and nuts.",
        "Get 7-8 hours of quality sleep every night.", "Replace refined carbs with whole grains.",
        "Avoid processed and packaged foods.", "Take short breaks and stretch during work hours.",
        "Incorporate low-impact cardio like swimming or cycling.", "Add more leafy greens to your diet.",
        "Reduce screen time before bedtime.", "Avoid smoking to improve heart health.",
        "Have regular health check-ups.", "Practice deep breathing exercises for relaxation.",
        "Include probiotic-rich foods like yogurt or kefir.", "Limit red meat consumption; opt for lean protein.",
        "Monitor your blood pressure regularly.", "Take prescribed medications on time.",
        "Avoid stressful situations.", "Include potassium-rich foods like bananas and spinach.",
        "Engage in daily brisk walking for 20 minutes."
    ],
    "Age_Group": [
        "30-50", "30-50", "All", "All", "All", "50+", 
        "All", "30-50", "All", "All", "All", "All", 
        "All", "All", "All", "All", "50+", "All", 
        "All", "All", "All", "All", "All", "All", "All"
    ],
    "BP_Category": [
        "Hypertension_Stage_1", "Normal_BP", "All", "All", "Hypertension_Stage_2", "Normal_BP",
        "All", "Normal_BP", "All", "All", "Hypertension_Stage_1", "All",
        "All", "All", "All", "Hypertension_Stage_2", "All", "All",
        "All", "All", "Hypertension_Stage_1", "Hypertension_Stage_2", "Hypertension_Stage_2",
        "Hypertension_Stage_1", "Hypertension_Stage_1"
    ],
    "Lifestyle": [
        "Sedentary", "Moderate", "All", "All", "All", "Active",
        "All", "Moderate", "All", "All", "Sedentary", "Sedentary",
        "Active", "All", "Sedentary", "All", "All", "Sedentary",
        "All", "Moderate", "Sedentary", "All", "Sedentary", "All", "All"
    ],
    "Emoji": [
        "🥗", "🧘", "🚶", "🌾", "☕❌", "🏋️",
        "💧", "🐟", "😴", "🍞", "📦❌", "🧍‍♀️",
        "🚴", "🥬", "📱❌", "🚬❌", "🩺", "🌬️",
        "🧃", "🍗❌", "📊", "💊", "😌", "🍌", "🏃"
    ]
}

# Step 1: Create the DataFrame
df = pd.DataFrame(data)

# Step 2: Encode categorical columns
def encode_categorical_data(df):
    df['Age_Group'] = df['Age_Group'].astype('category').cat.codes
    df['BP_Category'] = df['BP_Category'].astype('category').cat.codes
    df['Lifestyle'] = df['Lifestyle'].astype('category').cat.codes
    return df

df = encode_categorical_data(df)

# Step 3: Train-Test Split and Model Training
def train_model(df):
    # Target and Features
    y = df.index  # Use index as target to link back to the recommendation
    X_features = df[['Age_Group', 'BP_Category', 'Lifestyle']]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_features, y, test_size=0.2, random_state=42)

    # Train the Random Forest model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model

model = train_model(df)

# Step 4: Save the Model and Data
def save_model_and_data(model, df):
    joblib.dump(model, "recommendation_model.pkl")
    joblib.dump(df, "recommendation_data.pkl")

save_model_and_data(model, df)

# Step 5: Define a function to provide personalized recommendations
def get_personalized_recommendations(age, bp_category, lifestyle):
    # Encode inputs
    age_group = 0 if age < 30 else 1 if age <= 50 else 2
    bp_cat = 0 if bp_category == "Normal_BP" else 1
    life_style = {"Sedentary": 0, "Moderate": 1, "Active": 2}[lifestyle]

    # Create a DataFrame with proper feature names
    input_features = pd.DataFrame([[age_group, bp_cat, life_style]], columns=['Age_Group', 'BP_Category', 'Lifestyle'])
    
    # Predict recommendation index
    rec_index = model.predict(input_features)[0]

    # Fetch recommendation and emoji
    recommendation = df.loc[rec_index, "Recommendation"]
    emoji = df.loc[rec_index, "Emoji"]

    # Return recommendation with emoji
    return f"{recommendation} {emoji}"

# Example usage of the function
age = 35
bp_category = "Normal_BP"
lifestyle = "Active"
print(get_personalized_recommendations(age, bp_category, lifestyle))
