import pickle
from pathlib import Path
import pandas as pd
from flask import Flask, jsonify, render_template, request


# take helper function from the cricket_utils.py file.
from cricket_utils import (
    MAX_BALLS,      #120 balls
    clamp,          #To set the value between the min and max values
    get_innings_phase,   #To get the phase of the innings
    get_phase_score_margin,    #To score +- according to the phase.
    load_available_teams_from_csv,  
    parse_non_negative_int,   #To convert the user input into integer and validate them.
    parse_overs_to_balls,   # convert overs into balls.
    validate_selected_teams,  
)

FIRST_INNINGS_MODEL_PATH = Path("models/first_innings_model.pkl")
SECOND_INNINGS_MODEL_PATH = Path("models/second_innings_model.pkl")


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

# Flask application initialize kiya gaya hai.
app = Flask(__name__)


# This function load the saved ml model from the file.
def load_saved_model(model_path: Path):

    # model exist nhi karta hai to return or model empty hai to return
    if not model_path.exists() or model_path.stat().st_size == 0:
        return None

    try:
        # rb : read binary mode because file is in binary mode.
        with open(model_path, "rb") as file_handle:
            return pickle.load(file_handle)
        
        '''
        OSError → File is not open
        EOFError → File is corrupt
        UnpicklingError → Model is not in right format.
        '''    
    except (OSError, EOFError, pickle.UnpicklingError):
        return None

# Stored the trained model into the variables
FIRST_INNINGS_MODEL = load_saved_model(FIRST_INNINGS_MODEL_PATH)
SECOND_INNINGS_MODEL = load_saved_model(SECOND_INNINGS_MODEL_PATH)

# dataset se available teams load karke validation ke liye set me convert kiya gaya hai
AVAILABLE_TEAM_OPTIONS = load_available_teams_from_csv()
AVAILABLE_TEAM_SET = set(AVAILABLE_TEAM_OPTIONS)

# Agar CSV nahi mila → ye error show hoga
DATASET_UNAVAILABLE_ERROR = "Deliveries CSV dataset is not available. Make sure data/deliveries.csv exists."


# Input ko ML model ke liye ready banana
def create_first_innings_feature_frame(
    # This are the input parameters.
    batting_team: str,
    bowling_team: str,
    current_score: int,
    wickets_fallen: int,
    balls_bowled: int,
) -> pd.DataFrame:
    # agar sa ball dala hai to current run rate batayega or agar sa ball dala hi nhi hai to 0 batayega.
    current_run_rate = (current_score * 6 / balls_bowled) if balls_bowled else 0.0
    feature_values = {
        "batting_team": batting_team,
        "bowling_team": bowling_team,
        "current_score": current_score,
        "wickets_fallen": wickets_fallen,
        "balls_bowled": balls_bowled,
        "overs_completed": balls_bowled / 6.0,     #Ex : 63/6.0 = 10.3 over.
        "current_run_rate": current_run_rate,
    }
    return pd.DataFrame([feature_values], columns=FIRST_INNINGS_FEATURE_COLUMNS)


# Input ko ML model ke liye ready banana
def create_second_innings_feature_frame(
    batting_team: str,
    bowling_team: str,
    target_score: int,
    current_score: int,
    wickets_fallen: int,
    balls_bowled: int,
) -> tuple[pd.DataFrame, float]:      #this will return DataFrame for ML and the Required Run Rate for UI.
    runs_left = max(target_score - current_score, 0)
    balls_left = max(MAX_BALLS - balls_bowled, 0)
    wickets_left = max(10 - wickets_fallen, 0)
    current_run_rate = (current_score * 6 / balls_bowled) if balls_bowled else 0.0
    required_run_rate = (runs_left * 6 / balls_left) if balls_left else 0.0

    feature_values = {
        "batting_team": batting_team,
        "bowling_team": bowling_team,
        "target_score": target_score,
        "current_score": current_score,
        "wickets_fallen": wickets_fallen,
        "balls_bowled": balls_bowled,
        "runs_left": runs_left,
        "balls_left": balls_left,
        "wickets_left": wickets_left,
        "current_run_rate": current_run_rate,
        "required_run_rate": required_run_rate,
    }
    return pd.DataFrame([feature_values], columns=SECOND_INNINGS_FEATURE_COLUMNS), required_run_rate


@app.route("/")
def render_home_page():
    """Render the main page with the available team options."""
    return render_template("index.html", teams=AVAILABLE_TEAM_OPTIONS)

# This route predict the first inning score.
@app.route("/predict-first-innings", methods=["POST"])
def predict_first_innings_score():
    if FIRST_INNINGS_MODEL is None:
        return jsonify({"error": "First-innings model is not available. Run train_model.py first."}), 500
    if not AVAILABLE_TEAM_SET:
        return jsonify({"error": DATASET_UNAVAILABLE_ERROR}), 500

    try:
        # Frontend ka data safely lena
        request_data = request.get_json(silent=True) or {}

        # Sirf valid teams allow karna
        batting_team, bowling_team = validate_selected_teams(
            request_data.get("batting_team", ""),
            request_data.get("bowling_team", ""),
            available_teams=AVAILABLE_TEAM_SET,   
        )

        current_score = parse_non_negative_int(request_data.get("current_score"), "Current score")   # Current score.... is the error message if the error occured.

        wickets_fallen = parse_non_negative_int(
            request_data.get("wickets_fallen"),
            "Wickets fallen",
            maximum=10,
        )

        balls_bowled = parse_overs_to_balls(request_data.get("overs_completed"))

    except ValueError as validation_error:
        return jsonify({"error": str(validation_error)}), 400

    # ya decide karta hai ki match kounsa phase par hai start, middle or end.
    innings_phase = get_innings_phase(balls_bowled)

    #Check the innings is completed or not.
    if wickets_fallen == 10 or balls_bowled == MAX_BALLS:
        return jsonify(
            {
                "predicted_score": current_score,
                "lower_bound": current_score,
                "upper_bound": current_score,
                "phase": innings_phase,
                "innings_complete": True,
                "target_score": current_score + 1,
            }
        )

    first_innings_feature_frame = create_first_innings_feature_frame(
        batting_team,
        bowling_team,
        current_score,
        wickets_fallen,
        balls_bowled,
    )

    '''
    .predict() : function hai 
    first_innings_feature_frame : This is the input.
    [0] : list main sa nikalna quki output list main milta hai.
    '''
    # Model se score predict karo
    predicted_score = int(round(FIRST_INNINGS_MODEL.predict(first_innings_feature_frame)[0]))

    '''
    max() sirf ye ensure karta hai: "upper limit kabhi current score se kam na ho"
    Simple language me
    agar team already 280 bana chuki hai to tu usko 260 pe restrict nahi karega
    isliye:
        upper limit = max(current_score, 260)
    '''
    maximum_realistic_score = max(current_score, 260)


    '''
    clamp function ka use predicted score ko realistic range ke andar rakhne ke liye kiya gaya hai, taaki wo current score se kam ya unrealistic high na ho.
    '''
    # clamp(value, min ,max)
    predicted_score = int(clamp(predicted_score, current_score, maximum_realistic_score))


    '''
    ye line innings phase ke basis par prediction ke liye margin determine karti hai, jisse prediction ka range calculate kiya jata hai."
    '''
    # Early : 24 Middle : 15   Late : 8.    Also this was decided in the cricket_utils.py
    phase_score_margin = get_phase_score_margin(innings_phase)
    lower_bound = int(clamp(predicted_score - phase_score_margin, current_score, predicted_score))
    upper_bound = int(
        clamp(predicted_score + phase_score_margin, predicted_score, maximum_realistic_score)
    )

    # Final result frontend ko bhejna
    return jsonify(
        {
            "predicted_score": predicted_score,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "phase": innings_phase,
            "innings_complete": False,
            "target_score": None,
        }
    )

# second innings ke liye win probability calculate karne ke liye
@app.route("/predict-second-innings", methods=["POST"])
def predict_second_innings_win_probability():
    if SECOND_INNINGS_MODEL is None:
        return jsonify({"error": "Second-innings model is not available. Run train_model.py first."}), 500
    if not AVAILABLE_TEAM_SET:
        return jsonify({"error": DATASET_UNAVAILABLE_ERROR}), 500

    try:
        request_data = request.get_json(silent=True) or {}
        batting_team, bowling_team = validate_selected_teams(
            request_data.get("batting_team", ""),
            request_data.get("bowling_team", ""),
            available_teams=AVAILABLE_TEAM_SET,
        )
        target_score = parse_non_negative_int(request_data.get("target_score"), "Target score", minimum=1)
        current_score = parse_non_negative_int(request_data.get("current_score"), "Current score")
        wickets_fallen = parse_non_negative_int(
            request_data.get("wickets_fallen"),
            "Wickets fallen",
            maximum=10,
        )
        balls_bowled = parse_overs_to_balls(request_data.get("overs_completed"))
    except ValueError as validation_error:
        return jsonify({"error": str(validation_error)}), 400

    if current_score >= target_score:
        return jsonify(
            {
                "batting_team_win_pct": 100,
                "bowling_team_win_pct": 0,
                "required_run_rate": 0.0,
                "match_status": "Target reached",
            }
        )

    if wickets_fallen == 10:
        return jsonify(
            {
                "batting_team_win_pct": 0,
                "bowling_team_win_pct": 100,
                "required_run_rate": 0.0,
                "match_status": "Batting side all out",
            }
        )

    if balls_bowled == MAX_BALLS:
        return jsonify(
            {
                "batting_team_win_pct": 0,
                "bowling_team_win_pct": 100,
                "required_run_rate": 0.0,
                "match_status": "Overs complete",
            }
        )

    second_innings_feature_frame, required_run_rate = create_second_innings_feature_frame(
        batting_team,
        bowling_team,
        target_score,
        current_score,
        wickets_fallen,
        balls_bowled,
    )


    '''
    ML classification model ke andar classes hoti hain :  classes_ = [0, 1]    (0 : lose and 1 : Win)
    List ka andar win dhoon raha hai .index(1)  jo ki second position pa hai.

    '''
    # ye line model ke classes me se win class ka index nikalti hai
    chasing_team_win_class_index = list(SECOND_INNINGS_MODEL.classes_).index(1)

    batting_team_win_pct = int(
        round(
            clamp(
                # .predict_proba(second_innings_feature_frame) => [[0.3,0.7]] -> [0] => [0.3,0.7][1] => 0.7*100 => 70
                SECOND_INNINGS_MODEL.predict_proba(second_innings_feature_frame)[0][chasing_team_win_class_index] * 100,
                1,
                99,
            )
        )
    )
    bowling_team_win_pct = 100 - batting_team_win_pct

    return jsonify(
        {
            "batting_team_win_pct": batting_team_win_pct,
            "bowling_team_win_pct": bowling_team_win_pct,
            "required_run_rate": round(required_run_rate, 2),
            "match_status": "Live chase",
        }
    )


if __name__ == "__main__":
    app.run(debug=True, port=5000)
