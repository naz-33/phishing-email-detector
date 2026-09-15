import importlib.util
import numpy as np

spec = importlib.util.spec_from_file_location('appmod', 'app.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

assert mod.compute_prediction_confidence(0.90, 1) == 0.90
assert mod.compute_prediction_confidence(0.20, 0) == 0.80

class DummyTfidf:
    def transform(self, texts):
        return [[0.2, 0.8]] if 'urgent' in texts[0].lower() else [[0.8, 0.2]]

class DummyModel:
    def predict_proba(self, X):
        if X[0][1] > 0.5:
            return [[0.1, 0.9]]
        return [[0.8, 0.2]]

rep_phish = mod.predict_replication('urgent action required', DummyModel(), DummyTfidf())
assert rep_phish['prediction'] == 1
assert abs(rep_phish['phishing_probability'] - 0.9) < 1e-9
assert abs(rep_phish['confidence'] - 0.9) < 1e-9

rep_legit = mod.predict_replication('hello team meeting tomorrow', DummyModel(), DummyTfidf())
assert rep_legit['prediction'] == 0
assert abs(rep_legit['phishing_probability'] - 0.2) < 1e-9
assert abs(rep_legit['confidence'] - 0.8) < 1e-9

mod.preprocess_email_hybrid = lambda text: text.lower()
mod.extract_stylometrics = lambda text: np.array([100.0, 20.0, 0.1, 0.0, 0.0, 0.1, 0.0, 0.2, 1.0])

class DummyScaler:
    def transform(self, X):
        return np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

class DummyWordVec:
    def transform(self, X):
        from scipy.sparse import csr_matrix
        return csr_matrix([[1.0, 0.0]])

    def get_feature_names_out(self):
        return ['word_a']

class DummyCharVec:
    def transform(self, X):
        from scipy.sparse import csr_matrix
        return csr_matrix([[0.0, 1.0]])

    def get_feature_names_out(self):
        return ['char_a', 'char_b']

class DummyLR:
    n_features_in_ = 11
    def predict_proba(self, X):
        return [[0.1, 0.9]]

class DummyLGBM:
    n_features_in_ = 11
    feature_importances_ = np.array([0.1] * 11)
    def predict_proba(self, X):
        return [[0.1, 0.9]]

models = {
    'scaler': DummyScaler(),
    'word_vec': DummyWordVec(),
    'char_vec': DummyCharVec(),
    'lr_model': DummyLR(),
    'lgbm_model': DummyLGBM(),
    'config': {
        'ensemble_weights': {'logistic_regression': 0.6, 'lightgbm': 0.4},
        'threshold': 0.5,
    },
}

hyb = mod.predict_hybrid('urgent action required', models)
assert hyb['prediction'] == 1
assert abs(hyb['final_proba'] - 0.9) < 1e-9
assert abs(hyb['confidence'] - 0.9) < 1e-9

print('VALIDATION OK')
print('baseline phishing confidence =', rep_phish['confidence'])
print('baseline legitimate confidence =', rep_legit['confidence'])
print('hybrid phishing confidence =', hyb['confidence'])
