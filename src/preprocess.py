import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

def load_and_preprocess(path="data/ai4i2020.csv"):
    df = pd.read_csv(path)
    df = df.drop(columns=["UDI", "Product ID"])

    y = df["Machine failure"]
    X = df.drop(columns=["Machine failure", "TWF", "HDF", "PWF", "OSF", "RNF"])
    X = pd.get_dummies(X, columns=["Type"], drop_first=True)
    X.columns = X.columns.str.replace(r"[\[\]<]", "", regex=True)

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    smote = SMOTE(random_state=42)
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

    return X_train_bal, X_test, y_train_bal, y_test, scaler, X.columns.tolist()

