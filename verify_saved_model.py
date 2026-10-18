"""Run in the same directory as the saved, trusted model and app.py."""
from pathlib import Path
import joblib

model_path = Path(__file__).with_name('nigerian_language_identifier_svm.pkl')
if not model_path.exists():
    raise FileNotFoundError(f'Missing {model_path.name}; place it beside this script.')

# joblib/pickle files can execute code when loaded. Only load a model you created/trust.
model = joblib.load(model_path)
examples = [
    ('How are you today?', 'English'),
    ('Ẹ káàárọ̀', 'Yoruba'),
    ('Sannu da zuwa', 'Hausa'),
    ('Kedu ka ị mere', 'Igbo'),
]
for phrase, expected in examples:
    predicted = model.predict([phrase])[0]
    print(f'{phrase!r} -> {predicted} (expected {expected})')
print('Model loaded and predictions completed. Review outputs above.')
