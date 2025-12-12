import warnings
import pandas as pd
warnings.filterwarnings("ignore")
import seaborn as sns
import random
random.seed(42)

def deckCateg(pclass):
    if pclass == 1:
       return random.choice(['A', 'B', 'C'])
    elif pclass == 2:
       return random.choice(['D', 'E'])
    elif pclass == 3:
       return random.choice(['E', 'F', 'G'])
    return 'F'

titanic = sns.load_dataset('titanic')



#dropping redundant info
cleaned = titanic.drop(columns=['class', 'who', 'adult_male', 'embark_town', 'alive', 'alone'])
cleaned.drop_duplicates(inplace=True)
cleaned['embarked'].fillna(cleaned['embarked'].mode()[0], inplace=True)

cleaned['age'].fillna(cleaned['age'].mean(), inplace=True)
cleaned['sex'] = cleaned['sex'].map({'female':0, 'male': 1})

#deck
cleaned['deck'] = cleaned['deck'].astype('object')
missing_deck = cleaned['deck'].isna()
cleaned.loc[missing_deck, 'deck'] = cleaned.loc[missing_deck, 'pclass'].apply(deckCateg)
cleaned['deck'] = cleaned['deck'].astype('category')

#adding family size and isAlone columns
cleaned['FamilySize'] = cleaned['sibsp'] + cleaned['parch'] + 1
isAlone = lambda size: 0 if size > 1 else 1
cleaned['isAlone'] = cleaned['FamilySize'].apply(isAlone)

#cleaned.to_csv("/home/amine_pc/PyCharmMiscProject/Python M1/Python_Avance/Project_titanic/data/cleaned.csv")

#encoding categorial variables such as emarrked and deck
cleaned_one_hot = pd.get_dummies(
    cleaned,
    columns=['embarked', 'deck'],
    prefix=['embarked', 'deck']
)

#label encoding for the deck now data is ready for training
#cleaned_one_hot.to_csv("/home/amine_pc/PyCharmMiscProject/Python M1/Python_Avance/Project_titanic/data/one_hot.csv")