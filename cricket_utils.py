import csv
from pathlib import Path

# Convert old team name to new team name.
TEAM_ALIASES = {
    "Delhi Daredevils": "Delhi Capitals",
    "Kings XI Punjab": "Punjab Kings",
    "Rising Pune Supergiant": "Rising Pune Supergiants",
}

DELIVERIES_DATA_PATH = Path("data/deliveries.csv")
MAX_OVERS = 20
MAX_BALLS = MAX_OVERS * 6

# To make team name clead and standard.
'''
:str : string hint kar raha hai
-> str : this function return the string.
.get(key, default) : key dhoondo, agar mila to uski value do, nahi mila to default do

Case1 : match mil gaya
cleaned_name = "Delhi Daredevils"
outut : "Delhi Capitals"

Case2 : match nahi mila
cleaned_name = "Mumbai Indians"
output : "Mumbai Indians"
'''
def normalize_team_name(team_name: str) -> str:
    cleaned_name = str(team_name).strip()
    return TEAM_ALIASES.get(cleaned_name, cleaned_name)


# value ko limit ke andar rakhna.  lower bond <= value <= upper bound.
def clamp(value: float, lower_bound: float, upper_bound: float) -> float:
    return max(lower_bound, min(value, upper_bound))


# user input check karta hai
# ("120", "error message", minimum allowed value, maximum allowed value(None ka matlab abhi koi limit nhi))
def parse_non_negative_int(value, field_name: str, minimum: int = 0, maximum: int | None = None) -> int:
    """Parse and validate a whole-number input for score-related fields."""
    try:
        parsed_value = int(str(value).strip())
    except (TypeError, ValueError) as exc:
        # Current score must be a whole number
        raise ValueError(f"{field_name} must be a whole number.") from exc

    if parsed_value < minimum:
        # must be at least 0
        raise ValueError(f"{field_name} must be at least {minimum}.")
    
    if maximum is not None and parsed_value > maximum:
        raise ValueError(f"{field_name} cannot be more than {maximum}.")
    
    return parsed_value


# Overs (10.3) ko total balls me convert karna
def parse_overs_to_balls(value) -> int:
    overs_text = str(value).strip()
    if not overs_text:
        raise ValueError("Overs completed is required.")

    overs_parts = overs_text.split(".")    #"10.3" → ["10", "3"]
    if len(overs_parts) > 2:
        raise ValueError("Overs must use cricket notation like 10.3.")

    try:
        completed_overs = int(overs_parts[0])  #10
    except ValueError as exc:
        raise ValueError("Overs must be numeric.") from exc

    balls_in_current_over = 0

    '''
    len(overs_parts) == 2    :  kya input me dot (.) hai?
        Input	overs_parts	    len
        10.3	["10","3"]	    2 
        10	    ["10"]	        1 
    
    overs_parts[1] != ""     :  dot ke baad value empty to nahi

        Input	overs_parts[1]	    Result
        10.3	"3"	                Right
        10.	    ""	                Wrong
    '''
    # ye condition check karti hai ki overs input me dot ke baad valid balls value present ho, tabhi usko process kiya jata hai
    if len(overs_parts) == 2 and overs_parts[1] != "":    
        # ye block overs ke balls part ko integer me convert karta hai aur agar input invalid ho to custom error message throw karta hai
        try:
            balls_in_current_over = int(overs_parts[1])
        except ValueError as exc:
            raise ValueError("Overs must use cricket notation like 10.3.") from exc

    if completed_overs < 0 or balls_in_current_over < 0:
        raise ValueError("Overs cannot be negative.")
    if balls_in_current_over > 5:
        raise ValueError("Balls in an over must be between 0 and 5.")
    if completed_overs > MAX_OVERS or (completed_overs == MAX_OVERS and balls_in_current_over > 0):
        raise ValueError("Overs cannot exceed 20.")

    return (completed_overs * 6) + balls_in_current_over


def format_balls_as_overs_text(total_balls: int) -> str:
    """Convert a total ball count back into cricket overs notation."""
    return f"{total_balls // 6}.{total_balls % 6}"  #63//10 = 6 63%6 = 3   ->  10.3


def get_innings_phase(total_balls: int) -> str:
    """Map the innings progress to an early, middle, or late phase."""
    if total_balls <= 36:
        return "early"
    if total_balls <= 90:
        return "middle"
    return "late"


# innings phase ke hisaab se margin (± range) dena
def get_phase_score_margin(innings_phase: str) -> int:
    """Return the score margin used for the given innings phase."""
    phase_score_margins = {
        "early": 24,
        "middle": 15,
        "late": 8,
    }
    return phase_score_margins.get(innings_phase, 15)


# return type = list of strings (teams)
# ye function deliveries dataset se unique team names extract karta hai, unhe normalize karta hai aur sorted list ke form me return karta hai
def load_available_teams_from_csv() -> list[str]:
    """Load normalized team options directly from the deliveries CSV dataset."""
    if not DELIVERIES_DATA_PATH.exists():
        return []      #empty list return

    # : set[str]  : ye type hint hai matlab ye variable ek set hai aur isme string values hongi
    available_teams: set[str] = set()    

    try:
        with open(DELIVERIES_DATA_PATH, "r", encoding="utf-8", newline="") as csv_file:
            # har row ko dictionary bana deta hai
            deliveries_reader = csv.DictReader(csv_file)
            if not deliveries_reader.fieldnames:
                return []

            for delivery_row in deliveries_reader:
                batting_team_name = normalize_team_name(delivery_row.get("batting_team", ""))
                bowling_team_name = normalize_team_name(delivery_row.get("bowling_team", ""))

                if batting_team_name:
                    available_teams.add(batting_team_name)
                if bowling_team_name:
                    available_teams.add(bowling_team_name)
    except OSError:
        return []

    return sorted(available_teams)


# ye function ensure karta hai ki selected teams valid ho, empty na ho, ek dusre se different ho aur dataset me available ho
def validate_selected_teams(
    batting_team: str,
    bowling_team: str,
    available_teams: set[str] | None = None,
) -> tuple[str, str]:       #return karega (batting_team, bowling_team)
    normalized_batting = normalize_team_name(batting_team)
    normalized_bowling = normalize_team_name(bowling_team)

    if not normalized_batting or not normalized_bowling:
        raise ValueError("Select both batting and bowling teams.")
    if normalized_batting == normalized_bowling:
        raise ValueError("Batting and bowling teams must be different.")
    if available_teams is not None:
        if normalized_batting not in available_teams or normalized_bowling not in available_teams:
            raise ValueError("Selected teams must come from the deliveries CSV dataset.")

    return normalized_batting, normalized_bowling
