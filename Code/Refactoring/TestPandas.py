import pandas as pd
data = {"Name": [],
        "Month": [],
        "Year": [],
        "Sequence": []}
df = pd.DataFrame(data)
nameTest = ["A", "B", "C", "D", "E"]
monthTest = ["Jan", "Feb", "Mar", "Apr", "May"]
yearTest = [2020, 2021, 2022, 2023, 2024]
print(df)
for i in range(len(nameTest)):
    newRow = {"Name": [nameTest[i]], "Month": [monthTest[i]], "Year":[yearTest[i]]}
    newRowDF = pd.DataFrame(newRow)
    df = pd.concat([df, newRowDF], ignore_index=True)
    print(df)
df["Sequence"] = df["Sequence"].astype(str)
for idx, row in df.iterrows():
    df.loc[idx,"Sequence"] = df.loc[idx,"Name"] + str(df.loc[idx,"Year"])
print(df)
