import pandas as pd

data = {
    "home_team" : ["Brazil", "Argentina", "Germany", "France"],
    "away_team" : ["Crotia", "England", "Japan", "United States"],
    "home_score" : [2, 1, 0, 3],
    "away_score" : [1, 2, 2, 0]
}

df = pd.DataFrame(data)
#print(df)

#print(df["home_score"] > df["away_score"])

winner = df[df["home_score"] > df["away_score"]]
print(winner)

