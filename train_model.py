import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

def main():
    print("1. Loading PhiUSIIL dataset...")
    df = pd.read_csv("PhiUSIIL_Phishing_URL_Dataset.csv")
    
    # We select only robust, unbiased features to train the model.
    # We purposely exclude URL length and query-based features so the model 
    # doesn't falsely flag real URLs like Google as phishing.
    selected_features = ['DomainLength', 'IsDomainIP', 'NoOfSubDomain', 'IsHTTPS']
    
    X = df[selected_features]
    y = df['label'].astype(int).values
    feature_names = list(X.columns)

    print(f"Dataset shape: {X.shape[0]} samples with {len(feature_names)} robust features.")

    # Split dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("2. Training Random Forest classifier...")
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )
    clf.fit(X_train, y_train)
    
    print("\n--- Model Evaluation ---")
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))

    print("\n3. Exporting model artifacts...")
    artifacts = {
        "model": clf,
        "feature_names": feature_names
    }
    
    output_filename = "phishing_rf_model_hybrid.joblib"
    joblib.dump(artifacts, output_filename, compress=3)
    print(f"SUCCESS! Created '{output_filename}' ({len(feature_names)} features saved).")

if __name__ == "__main__":
    main()
