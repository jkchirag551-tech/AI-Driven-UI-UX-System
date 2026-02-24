import pandas as pd
import random
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier

CATEGORIES = ['Automotive', 'Beauty', 'Construction', 'Cybersecurity', 'Education', 'Entertainment', 'Fashion', 'Finance', 'Food', 'Gaming', 'Healthcare', 'Interior', 'Legal', 'Marketing', 'Music', 'Non-Profit', 'Photography', 'Real Estate', 'Sports', 'Technology', 'Travel']
VIBES = ['Commercial', 'Promotional', 'Balanced', 'Corporate', 'Professional']

# Generating dummy training data for the brain
data = []
for c_idx, cat in enumerate(CATEGORIES):
    for v_idx, vibe in enumerate(VIBES):
        for var in range(5):
            layout = 'Bento Grid' if v_idx > 2 else 'Split Screen'
            font = 'Inter' if v_idx > 2 else 'Montserrat'
            p, s, t = ('#0f172a', '#f8fafc', '#3b82f6') if v_idx > 2 else ('#ff0055', '#1a1a1a', '#ffcc00')
            data.append([c_idx, v_idx, var, layout, font, p, s, t])

df = pd.DataFrame(data, columns=['Cat_Code', 'Vibe_Code', 'Variation', 'Layout', 'Font', 'Primary', 'Secondary', 'Tertiary'])
brain = MultiOutputClassifier(RandomForestClassifier())
brain.fit(df[['Cat_Code', 'Vibe_Code', 'Variation']], df[['Layout', 'Font', 'Primary', 'Secondary', 'Tertiary']])

with open('model_brain.pkl', 'wb') as f:
    pickle.dump(brain, f)
print("Brain trained successfully!")