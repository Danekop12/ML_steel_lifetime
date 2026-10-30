import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
import sklearn
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler, MinMaxScaler

Steel_Data = pd.read_excel("../data/NIMS_FATIGUE.xlsx")
Steel_Data.columns = Steel_Data.columns.str.strip()

# Summarize data
pd.set_option("display.width", 300)
pd.set_option("display.max_columns", None) # to show all columns
print(Steel_Data.describe())

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

##########################
# Kjøre OneHotEncoding på kolonne "Testing Material" - gir egne kolonner for hver kategori - med binære verdier i kolonnene
encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
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

###########################
# The attributes of the dataset regarding temperature are "Normalizing temp", "Quenching temp" and "Tempering temp".
# These attributes have quite a few rows with missing values, and it is considered informative to retain information about their presence or absence.
# Due to the fact that these attributes are related to heat treatment processes, the presence or absence of these values may provide valuable insights for the model.
# Therefore, we will create new columns that indicate whether the original columns have values or not. This way, the model can learn from the presence or absence of these values.

# First the 3 new columns will be added to the dataset - and filled with 1 if the original column has a value (notna=True), and 0 if it is missing (notna=False).
Steel_Data3["normalizing_temp_present"] = Steel_Data3["Normalizing temp"].notna().astype(int)
Steel_Data3["quenching_temp_present"] = Steel_Data3["Quenching temp"].notna().astype(int)
Steel_Data3["tempering_temp_present"] = Steel_Data3["Tempering temp"].notna().astype(int)

# Secondly the missing values in the original columns will be filled with 0, as the presence or absence of these values is already captured in the new columns.
# It is not considered appropriate to fill the missing values with the mean or median, as this would introduce information about treatments on the metal that have not actually been performed.
Steel_Data3["Normalizing temp"] = Steel_Data3["Normalizing temp"].fillna(0)
Steel_Data3["Quenching temp"] = Steel_Data3["Quenching temp"].fillna(0)
Steel_Data3["Tempering temp"] = Steel_Data3["Tempering temp"].fillna(0)

print(Steel_Data3.head())

##############################
# Run correlation analysis of columns to identify those that are highly correlated with each other.
# This can help reduce multicollinearity in the model and improve its performance.
# Running Pandas corr() function with Spearman method.
corr_matrix = Steel_Data3.corr(method="spearman", numeric_only=True)

upper = corr_matrix.where(
    np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

high_corr = (
    upper.stack()
         .reset_index()
    )

high_corr.columns = ["Feature1", "Feature2", "Correlation"]

high_corr = high_corr.reindex(
    high_corr["Correlation"].abs().sort_values(ascending=False).index
    )

print("------------------------------------------")
print(high_corr.head(20))
#print(high_corr.to_string())


###############################
# Choosing which Standardize to use
scaler = StandardScaler() # No fixed bounds, sensetive to extreme outliers. No fixed range
#scaler = MinMaxScaler() # Opposite situation, where all values could be too squeezed of outlier. Range [0,1]
#scaler = RobustScaler() # If we gonna try to use all values, this one could work better. No fixed range

Steel_Data_Scaled = Steel_Data.copy()
# List of all columns to standardize
columns = ["Reduction Ratio", "Normalizing temp", "Quenching temp", "Tempering temp", "0.2% proof stress",
        "Upper yield stress", "Tensile strength", "True fracture stress", "Vickers hardness", "σa"]
print(Steel_Data_Scaled[columns].head())
Steel_Data_Scaled[columns] = scaler.fit_transform(Steel_Data_Scaled[columns])



print("------------------------------------------")
print(Steel_Data_Scaled.describe())