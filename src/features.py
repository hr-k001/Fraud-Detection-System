from sklearn.preprocessing import LabelEncoder

class FeatureEngineer:
    def __init__(self):
        self.encoders = {}

    def fit(self, df, cat_cols):
        for col in cat_cols:
            le = LabelEncoder()
            df[col] = df[col].astype(str)
            le.fit(df[col])
            self.encoders[col] = le
        return self

    def transform(self, df):
        df = df.copy()

        for col, le in self.encoders.items():
            if col in df.columns:
                df[col] = df[col].astype(str)
                df[col] = df[col].map(
                    lambda x: le.transform([x])[0] if x in le.classes_ else -1
                )

        return df