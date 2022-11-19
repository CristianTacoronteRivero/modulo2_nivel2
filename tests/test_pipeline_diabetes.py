from nivel2.pipeline_diabetes import Pipeline_diabetes
import pandas as pd

def test_select_features_DataDrame_Series():
    df = pd.read_csv('https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.csv')
    cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigree', 'Age', 'Outcome']
    diabetes = Pipeline_diabetes(dataframe=df,
                    cols=cols,
                    seed=42)

    X, y = diabetes.select_features(target='Outcome')

    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
