@echo off
git add -f "static/models/chronic_kidney_disease/data/processed_kidney_disease.csv"
git add -f "static/models/chronic_kidney_disease/Random_Forest_model.pkl"
git add -f "static/models/lung_disease/model.h5"
git add ".gitignore"
git add "fix_csv.py"
git status --short
echo --- staging done ---
