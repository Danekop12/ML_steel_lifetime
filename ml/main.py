import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
import sklearn
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler, MinMaxScaler

Steel_Data = pd.read_excel("../data/NIMS_FATIGUE.xlsx")

# Tester konvertering til log10 av Nf verdiene i datasettet. Legger log10 verdiene som egen kolonne i datasettet
Steel_Data["Nf_log"] = np.log10(Steel_Data["Nf"])

print(Steel_Data.head())

# skriver ut oversikt over kolonner/rader med null verdier ### Må avgjøre hva som skal gjøres med disse??
print(Steel_Data.isnull().sum())

# sjekker for  og skriver ut duplikate rader #### Hva gjør vi med alle radene som er like?? og hvordan identifisere hvilke som er like
duplikat_rader = Steel_Data[Steel_Data.duplicated(keep=False)]
print(duplikat_rader) # oppgis å være 615 duplikate rader (men da flere ulike duplikater)

# Sletter alle duplikate rader - beholder bare første forekomst 
Steel_Data = Steel_Data.drop_duplicates(keep='first')
print(Steel_Data.info())
print(Steel_Data)      ##### Ser ut som det bare er slettet 2-3 rader. Hvorfor? Eller er det bare indeks som er beholdt

# Fjerner kolonne "Heat" og "DS No. PDF"  ##### Må kunne begrunne hvorfor? Kan ev. kjøre modellen senere med Heat for å sjekke om resultatet blir bedre
Steel_Data.drop(columns=['Heat', 'DS No. PDF'], inplace=True)
print(Steel_Data.head())
print(Steel_Data.info())


# Kjøre OneHotEncoding på kolonne "Testing Material" - gir egne kolonner for hver kategori - med binære verdier i kolonnene
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
testing_material_encoded = encoder.fit_transform(Steel_Data[['Testing Material']])
testing_material_encoded_df = pd.DataFrame(testing_material_encoded, columns=encoder.get_feature_names_out(['Testing Material']))
Steel_Data2 = pd.concat([Steel_Data, testing_material_encoded_df], axis=1)
Steel_Data2.drop(columns=['Testing Material'], inplace=True)
print(Steel_Data2.head())

# Kjøre OneHotEncoding på kolonne "Type of test" - gir egne kolonner for hver kategori - med binære verdier i kolonnene
type_of_test_encoded = encoder.fit_transform(Steel_Data[['Type of test']])
type_of_test_encoded_df = pd.DataFrame(type_of_test_encoded, columns=encoder.get_feature_names_out(['Type of test']))
Steel_Data3 = pd.concat([Steel_Data2, type_of_test_encoded_df], axis=1)
Steel_Data3.drop(columns=['Type of test'], inplace=True)
print(Steel_Data3.head())
print(Steel_Data.head())