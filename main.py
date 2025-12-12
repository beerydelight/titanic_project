from Utils.Outlier_handler import outlier_handled
from Utils.utils import titanic
from Utils.utils import cleaned_one_hot

import seaborn as sns
import matplotlib.pyplot as plt

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    #analyse des données
    titanic.head()

    #extraction des valeurs null
    titanic.isnull().sum()

    #Analyse graphique (correlation avec les donnée)
    sns.catplot(x=titanic["sex"], hue=titanic["survived"],kind="count", data=titanic)

    # Group the dataset by Pclass and Survived and then unstack them
    group = titanic.groupby(['Pclass', 'Survived'])
    pclass_survived = group.size().unstack()

    # Heatmap - Color encoded 2D representation of data.
    sns.heatmap(pclass_survived, annot=True, fmt="d")

    # Violinplot Displays distribution of data
    # across all levels of a category.
    sns.violinplot(x="Sex", y="Age", hue="Survived", data=titanic, split=True)

    # Countplot
    sns.catplot(x='Embarked', hue='Survived',kind='count', col='Pclass', data=titanic)
    plt.show()

