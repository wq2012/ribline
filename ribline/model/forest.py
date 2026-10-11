import os
import joblib
from sklearn.ensemble import RandomForestClassifier

class RibForest:
    def __init__(self, n_estimators=100, max_depth=None, n_jobs=-1, random_state=42):
        self.clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            n_jobs=n_jobs,
            random_state=random_state,
            class_weight='balanced' # Ribs are sparse
        )
        
    def fit(self, X, y):
        self.clf.fit(X, y)
        
    def predict(self, X):
        return self.clf.predict(X)
        
    def predict_proba(self, X):
        return self.clf.predict_proba(X)
        
    def save(self, filepath):
        joblib.dump(self.clf, filepath)

    def load(self_or_filepath, filepath=None):
        """Loads a saved RandomForestClassifier into a RibForest instance.

        Can be called either on an instance (`rf.load(path)`) or directly on the
        class (`rf = RibForest.load(path)`).
        """
        if isinstance(self_or_filepath, RibForest):
            if filepath is None:
                raise TypeError("RibForest.load() missing required argument: 'filepath'")
            self_or_filepath.clf = joblib.load(filepath)
            return self_or_filepath
        else:
            instance = RibForest()
            instance.clf = joblib.load(self_or_filepath)
            return instance

    @classmethod
    def from_path(cls, filepath):
        instance = cls()
        instance.load(filepath)
        return instance

