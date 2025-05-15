
from pathlib import Path
import pandas as pd


def load_resume_dataset(path):
    """
    Load the resume dataset from the given path.
    """
    return pd.read_csv(path)


if __name__ == "__main__":
    path = Path(__file__).parent.parent / "data" / "Resume.csv"
    resume_dataset = load_resume_dataset(path)
    print(resume_dataset.head())
    print(resume_dataset.columns)
    print(resume_dataset.shape)
    print(resume_dataset.info())
    print(resume_dataset.describe())
    print(resume_dataset.isnull().sum())
    print(resume_dataset.duplicated().sum())
    print(resume_dataset.nunique())
    print(resume_dataset["Resume_str"])
