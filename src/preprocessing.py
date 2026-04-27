class DataPreprocessor:
    def __init__(self, cols_to_drop, num_cols, cat_cols):
        self.cols_to_drop = cols_to_drop
        self.num_cols = num_cols
        self.cat_cols = cat_cols
        self.medians = {}

    def fit(self, df):
        for col in self.num_cols:
            self.medians[col] = df[col].median()
        return self

    def transform(self, df):
        df = df.copy()

        df = df.drop(columns=self.cols_to_drop, errors='ignore')

        for col in self.num_cols:
            if col in df.columns:
                df[col] = df[col].fillna(self.medians[col])

        for col in self.cat_cols:
            if col in df.columns:
                df[col] = df[col].fillna('missing')

        return df