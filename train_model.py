import pickle
from pathlib import Path
import pandas as pd

from sklearn.compose import ColumnTransformer    #Different columns par different preprocessing apply karne ke liye.
from sklearn.ensemble import RandomForestRegressor  
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, mean_absolute_error, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


from cricket_utils import MAX_BALLS, normalize_team_name

DELIVERIES_DATA_PATH = Path("data/deliveries.csv")
MODEL_DIR = Path("models")

FIRST_INNINGS_MODEL_PATH = MODEL_DIR / "first_innings_model.pkl"
SECOND_INNINGS_MODEL_PATH = MODEL_DIR / "second_innings_model.pkl"


# Model ko prediction ke liye fixed features chahiye hote hain, isliye feature columns predefined rakhe gaye hain.
FIRST_INNINGS_FEATURE_COLUMNS = [
    "batting_team",
    "bowling_team",
    "current_score",
    "wickets_fallen",
    "balls_bowled",
    "overs_completed",
    "current_run_rate",
]

# Model ko prediction ke liye fixed features chahiye hote hain, isliye feature columns predefined rakhe gaye hain.
SECOND_INNINGS_FEATURE_COLUMNS = [
    "batting_team",
    "bowling_team",
    "target_score",
    "current_score",
    "wickets_fallen",
    "balls_bowled",
    "runs_left",
    "balls_left",
    "wickets_left",
    "current_run_rate",
    "required_run_rate",
]

# ye function deliveries dataset ko process karke do DataFrames return karta hai — ek complete cleaned data aur doosra sirf legal ball snapshots jisme important features calculate kiye gaye hote hain
def load_deliveries_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    # sirf important columns load karega unnecessary data ignore
    required_columns = [
        "match_id",
        "inning",
        "batting_team",
        "bowling_team",
        "over",
        "ball",
        "wide_runs",
        "noball_runs",
        "total_runs",
        "player_dismissed",
        "is_super_over",
    ]

    # CSV load
    deliveries_frame = pd.read_csv(DELIVERIES_DATA_PATH, usecols=required_columns)

    # team names clean for batting team
    deliveries_frame["batting_team"] = deliveries_frame["batting_team"].map(normalize_team_name)
    
    # team names clean for bowling team
    deliveries_frame["bowling_team"] = deliveries_frame["bowling_team"].map(normalize_team_name)

    '''
    .notna() : kuch likha hai ya empty hai
    .astype(int) : True/False ko number banata hai -> 1/0
    '''
    # ye line player_dismissed column ko check karke ek binary column create karti hai jahan wicket hone par 1 aur na hone par 0 assign kiya jata hai. Agar player out hua → 1, nahi hua → 0
    deliveries_frame["is_wicket"] = deliveries_frame["player_dismissed"].notna().astype(int)

    '''
    Agar sa value > 0 hai to iska matlab no ball ya wide hoga
    or == 0 hai to wide ya no ball nhi hoga.
    '''
    # ye line check karti hai ki ball legal hai ya nahi, jahan wide aur no-ball na hone par 1 assign kiya jata hai
    deliveries_frame["is_legal_ball"] = (
        (deliveries_frame["wide_runs"].fillna(0) == 0)
        & (deliveries_frame["noball_runs"].fillna(0) == 0)
    ).astype(int)

    # copy() ka use ek independent DataFrame banane ke liye kiya jata hai taaki original data accidentally modify na ho
    # is line me super over data ko remove kiya gaya hai.
    deliveries_frame = deliveries_frame[deliveries_frame["is_super_over"] == 0].copy()

    # is line me sirf main innings ka data retain kiya gaya hai
    # column.isin([values]) : kya column ki value in values me se kisi se match karti hai
    deliveries_frame = deliveries_frame[deliveries_frame["inning"].isin([1, 2])].copy()

    # data ko match, inning, over aur ball ke chronological order me arrange kiya gaya hai taaki match progression sahi tarike se reflect ho
    # reset_index(drop=True) : index ko fresh bana raha
    deliveries_frame = deliveries_frame.sort_values(["match_id", "inning", "over", "ball"]).reset_index(drop=True)

    # is logic me sirf un matches ko retain kiya gaya hai jisme dono innings available ho, taaki model training ke liye complete data mile.
    # match1 = [1,2] match2 = [1] ...
    innings_count_per_match = deliveries_frame.groupby("match_id")["inning"].nunique()
    # Jaha pa do inning hai usa hi return karega.
    complete_match_ids = innings_count_per_match[innings_count_per_match == 2].index
    # Do inning wala match ko hi allow karega model ka liya.
    deliveries_frame = deliveries_frame[deliveries_frame["match_id"].isin(complete_match_ids)].copy()

    # cumsum() : 1-1-2-3-5-8.....
    # groupby ka use karke har match aur inning ke liye running total independently calculate kiya gaya hai
    deliveries_frame["balls_bowled"] = deliveries_frame.groupby(["match_id", "inning"])["is_legal_ball"].cumsum()
    deliveries_frame["current_score"] = deliveries_frame.groupby(["match_id", "inning"])["total_runs"].cumsum()
    deliveries_frame["wickets_fallen"] = deliveries_frame.groupby(["match_id", "inning"])["is_wicket"].cumsum()

    # sirf valid balls ka data lena + overs aur run rate calculate karna
    # extract only legal balls.
    legal_ball_snapshots = deliveries_frame[deliveries_frame["is_legal_ball"] == 1].copy()
    # convert into overs
    legal_ball_snapshots["overs_completed"] = legal_ball_snapshots["balls_bowled"] / 6.0
    # get the runrate
    legal_ball_snapshots["current_run_rate"] = (
        legal_ball_snapshots["current_score"] * 6 / legal_ball_snapshots["balls_bowled"]
    )

    return deliveries_frame, legal_ball_snapshots

# For every match to start with the zero
def build_starting_snapshots(innings_frame: pd.DataFrame) -> pd.DataFrame:
    """Add a zero-score starting state so the models can learn from ball zero."""
    starting_snapshots = innings_frame.groupby("match_id", as_index=False).first().copy()
    starting_snapshots["balls_bowled"] = 0
    starting_snapshots["current_score"] = 0
    starting_snapshots["wickets_fallen"] = 0
    starting_snapshots["overs_completed"] = 0.0
    starting_snapshots["current_run_rate"] = 0.0
    return starting_snapshots


def build_first_innings_training_dataset(
    deliveries_frame: pd.DataFrame,
    legal_ball_snapshots: pd.DataFrame,
) -> pd.DataFrame:
    # only fetching the first innings score.
    first_innings_snapshots = legal_ball_snapshots[legal_ball_snapshots["inning"] == 1].copy()
    # har match ka first innings final score nikaal raha hai
    final_first_innings_scores = (
        deliveries_frame[deliveries_frame["inning"] == 1]
        .groupby("match_id")["current_score"]
        .max()
        .rename("final_score")
        .reset_index()
    )

    # har match ka 0 score wala starting point add
    starting_snapshots = build_starting_snapshots(first_innings_snapshots)

    first_innings_snapshots = pd.concat([first_innings_snapshots, starting_snapshots], ignore_index=True, sort=False)
    first_innings_snapshots = first_innings_snapshots.merge(final_first_innings_scores, on="match_id", how="left")
    return first_innings_snapshots


def build_second_innings_training_dataset(
    deliveries_frame: pd.DataFrame,
    legal_ball_snapshots: pd.DataFrame,
) -> pd.DataFrame:
    """Create the training rows used to predict second-innings chase outcomes."""
    first_innings_total_scores = (
        deliveries_frame[deliveries_frame["inning"] == 1]
        .groupby("match_id")["current_score"]
        .max()
        .rename("first_innings_total")
        .reset_index()
    )

    second_innings_snapshots = legal_ball_snapshots[legal_ball_snapshots["inning"] == 2].copy()
    second_innings_snapshots = second_innings_snapshots.merge(first_innings_total_scores, on="match_id", how="left")
    second_innings_snapshots["target_score"] = second_innings_snapshots["first_innings_total"] + 1

    final_second_innings_scores = (
        deliveries_frame[deliveries_frame["inning"] == 2]
        .groupby("match_id")["current_score"]
        .max()
        .rename("second_innings_total")
        .reset_index()
    )
    second_innings_snapshots = second_innings_snapshots.merge(final_second_innings_scores, on="match_id", how="left")
    second_innings_snapshots["chasing_team_won"] = (
        second_innings_snapshots["second_innings_total"] >= second_innings_snapshots["target_score"]
    ).astype(int)

    starting_snapshots = build_starting_snapshots(second_innings_snapshots)
    starting_snapshots["first_innings_total"] = starting_snapshots["first_innings_total"]
    starting_snapshots["target_score"] = starting_snapshots["first_innings_total"] + 1
    starting_snapshots["second_innings_total"] = starting_snapshots["second_innings_total"]
    starting_snapshots["chasing_team_won"] = starting_snapshots["chasing_team_won"]

    second_innings_snapshots = pd.concat([second_innings_snapshots, starting_snapshots], ignore_index=True, sort=False)
    second_innings_snapshots["runs_left"] = (second_innings_snapshots["target_score"] - second_innings_snapshots["current_score"]).clip(
        lower=0
    )
    second_innings_snapshots["balls_left"] = (MAX_BALLS - second_innings_snapshots["balls_bowled"]).clip(lower=0)
    second_innings_snapshots["wickets_left"] = (10 - second_innings_snapshots["wickets_fallen"]).clip(lower=0)
    second_innings_snapshots["required_run_rate"] = 0.0

    live_chase_mask = second_innings_snapshots["balls_left"] > 0
    second_innings_snapshots.loc[live_chase_mask, "required_run_rate"] = (
        second_innings_snapshots.loc[live_chase_mask, "runs_left"] * 6 / second_innings_snapshots.loc[live_chase_mask, "balls_left"]
    )

    second_innings_snapshots = second_innings_snapshots[
        (second_innings_snapshots["runs_left"] > 0)
        & (second_innings_snapshots["balls_left"] > 0)
        & (second_innings_snapshots["wickets_left"] > 0)
    ].copy()
    return second_innings_snapshots


def split_match_ids_for_training(match_ids: pd.Series) -> tuple[set[int], set[int]]:
    """Split matches into train and test groups without leaking the same match."""
    unique_matches = pd.DataFrame({"match_id": sorted(match_ids.unique())})

    # Divide the data into train and test. 
    match_splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    # next() only takes the first split only.
    train_match_index, test_match_index = next(
        match_splitter.split(unique_matches, groups=unique_matches["match_id"])
    )
    training_match_ids = set(unique_matches.iloc[train_match_index]["match_id"].tolist())
    testing_match_ids = set(unique_matches.iloc[test_match_index]["match_id"].tolist())
    return training_match_ids, testing_match_ids


# First innings score predict ml pipeline.
def create_first_innings_model_pipeline() -> Pipeline:
    """Build the regression pipeline used for first-innings score prediction."""
    categorical_features = ["batting_team", "bowling_team"]
    numeric_features = [column for column in FIRST_INNINGS_FEATURE_COLUMNS if column not in categorical_features]

    # ColumnTransformer : Different columns par different preprocessing apply karna.
    preprocessor = ColumnTransformer(
        transformers=[
            # OneHotEncoder : Text ko numbers me convert karta hai.
            ("teams", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("numbers", "passthrough", numeric_features),
        ]
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=90,  #90 decision trees use karo
                    random_state=42,   
                    n_jobs=-1,     #CPU ke saare cores use karo.
                    max_depth=18,
                    min_samples_leaf=3,
                ),
            ),
        ]
    )

# Second innings win prediction model pipeline banata hai
def create_second_innings_model_pipeline() -> Pipeline:
    """Build the classification pipeline used for second-innings win prediction."""
    categorical_features = ["batting_team", "bowling_team"]
    numeric_features = [column for column in SECOND_INNINGS_FEATURE_COLUMNS if column not in categorical_features]

    preprocessor = ColumnTransformer(
        transformers=[
            ("teams", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            # Numbers ko same scale/range me convert karta hai.  180,2   ==  0.8,-0.5
            ("numbers", StandardScaler(), numeric_features),
        ]
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1200)),    #1200 attempts tak learn karta hai.
        ]
    )

# To evaluate first innings model accuracy.
def evaluate_first_innings_model(model: Pipeline, test_frame: pd.DataFrame) -> float:
    """Measure first-innings prediction error with mean absolute error."""
    # model.predict ml model predection karta hai
    predictions = model.predict(test_frame[FIRST_INNINGS_FEATURE_COLUMNS])
    return mean_absolute_error(test_frame["final_score"], predictions)


def evaluate_second_innings_model(model: Pipeline, test_frame: pd.DataFrame) -> tuple[float, float]:
    """Measure second-innings model quality with accuracy and ROC-AUC."""
    predictions = model.predict(test_frame[SECOND_INNINGS_FEATURE_COLUMNS])
    probabilities = model.predict_proba(test_frame[SECOND_INNINGS_FEATURE_COLUMNS])[:, 1]   #1 represent win probability.
    classification_accuracy = accuracy_score(test_frame["chasing_team_won"], predictions)  #Kitni predictions sahi tha classification_accuracy check karta hai. 
    try:
        # roc_auc_score() : Model probabilities kitni achi hai measure karta hai.
        classification_roc_auc = roc_auc_score(test_frame["chasing_team_won"], probabilities)
    except ValueError:
        classification_roc_auc = 0.0
    return classification_accuracy, classification_roc_auc


def main() -> None:
    """Train both models, save the artifacts, and print evaluation details."""
    # Agar parent folders missing ho vo bhi bana do that is models and Agar folder already exist kare error mat do
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    if not DELIVERIES_DATA_PATH.exists():
        print(f"Error: {DELIVERIES_DATA_PATH} was not found.")
        return

    print("Loading deliveries dataset...")
    deliveries_frame, legal_ball_snapshots = load_deliveries_data()
    first_innings_training_frame = build_first_innings_training_dataset(deliveries_frame, legal_ball_snapshots)
    second_innings_training_frame = build_second_innings_training_dataset(deliveries_frame, legal_ball_snapshots)

    training_match_ids, testing_match_ids = split_match_ids_for_training(first_innings_training_frame["match_id"])

    first_innings_train_frame = first_innings_training_frame[first_innings_training_frame["match_id"].isin(training_match_ids)].copy()
    first_innings_test_frame = first_innings_training_frame[first_innings_training_frame["match_id"].isin(testing_match_ids)].copy()
    second_innings_train_frame = second_innings_training_frame[second_innings_training_frame["match_id"].isin(training_match_ids)].copy()
    second_innings_test_frame = second_innings_training_frame[second_innings_training_frame["match_id"].isin(testing_match_ids)].copy()

    print("Training first-innings score model...")
    first_innings_model = create_first_innings_model_pipeline()
    # fit() : ml start the training to learn the pattern
    first_innings_model.fit(first_innings_train_frame[FIRST_INNINGS_FEATURE_COLUMNS], first_innings_train_frame["final_score"])

    print("Training second-innings win probability model...")
    second_innings_model = create_second_innings_model_pipeline()
    second_innings_model.fit(second_innings_train_frame[SECOND_INNINGS_FEATURE_COLUMNS], second_innings_train_frame["chasing_team_won"])

    first_innings_mae = evaluate_first_innings_model(first_innings_model, first_innings_test_frame)
    second_innings_accuracy, second_innings_roc_auc = evaluate_second_innings_model(second_innings_model, second_innings_test_frame)

# wb write binary mode.
    with open(FIRST_INNINGS_MODEL_PATH, "wb") as file_handle:
        pickle.dump(first_innings_model, file_handle)

    with open(SECOND_INNINGS_MODEL_PATH, "wb") as file_handle:
        pickle.dump(second_innings_model, file_handle)

    print(f"Saved first-innings model to {FIRST_INNINGS_MODEL_PATH}")
    print(f"Saved second-innings model to {SECOND_INNINGS_MODEL_PATH}")
    print(f"First innings MAE: {first_innings_mae:.2f}")
    print(f"Second innings accuracy: {second_innings_accuracy:.3f}")
    print(f"Second innings ROC-AUC: {second_innings_roc_auc:.3f}")


if __name__ == "__main__":
    main()

