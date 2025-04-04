# AY24-25-QF5214-Group-9
## 📥 Downloading Data

# The large data file `Panel Data of Monthly Frequency Regression with Adjusted Sorting.csv` is stored using git lfs.
To access the file:
1. Navigate to git-lfs.com and click Download.
2. On your computer, locate the downloaded file.
3. Double click on the file called git-lfs-windows-1.X.X.exe, where 1.X.X is replaced with the Git LFS version you downloaded. When you open this file Windows will run a setup wizard to install Git LFS.
4. Clone the Repository with Git LFS Support:

```
git clone https://github.com/MRLMaorong/AY24-25-QF5214-Group-9.git
cd AY24-25-QF5214-Group-9/Data
git lfs install
git lfs pull
```

# The large data file `factor_second.pkl` is not stored in the repository due to size limitations.

To download it, run:

```
pip install gdown
python Data/download_model.py
```
