"""Demo script: predict a handful of knockout ties from the saved model.

Kept as a quick way to sanity-check a freshly trained bundle. The same thing is
available from the command line:

    wcp predict Belgium Spain --knockout
"""

from wcpredictor import Predictor

# (home, away, neutral venue?)
MATCHES = [
    ("Belgium", "Spain", True),
    ("England", "Norway", True),
    ("Argentina", "Switzerland", True),
    ("Switzerland", "Argentina", True),
]


def main() -> None:
    predictor = Predictor.load()
    for home, away, neutral in MATCHES:
        prediction = predictor.predict(
            home,
            away,
            neutral=neutral,
            competition="major",
            knockout=True,
        )
        print(prediction.format())
        print()


if __name__ == "__main__":
    main()
