const battingTeamSelect = document.getElementById("batting_team");
const bowlingTeamSelect = document.getElementById("bowling_team");
const teamSelectionError = document.getElementById("team_error");

const firstInningsForm = document.getElementById("first_innings_form");
const secondInningsForm = document.getElementById("second_innings_form");
const firstInningsFormError = document.getElementById("first_form_error");
const secondInningsFormError = document.getElementById("second_form_error");
const targetScoreField = document.getElementById("target_score");
const firstInningsTeamSummary = document.getElementById("first_innings_teams");
const secondInningsTeamSummary = document.getElementById("second_innings_teams");

const predictedScoreValueElement = document.getElementById("predicted_score_value");
const predictedScoreRangeElement = document.getElementById("predicted_score_range");
const battingTeamLabelElement = document.getElementById("batting_team_label");
const bowlingTeamLabelElement = document.getElementById("bowling_team_label");
const battingTeamWinPercentageElement = document.getElementById("batting_win_pct");
const bowlingTeamWinPercentageElement = document.getElementById("bowling_win_pct");
const battingTeamWinBarElement = document.getElementById("batting_win_bar");
const bowlingTeamWinBarElement = document.getElementById("bowling_win_bar");
const winProbabilityMetaElement = document.getElementById("win_probability_meta");

// This function shows or hides an error message for the given UI element.
function updateErrorMessage(errorElement, errorMessage) {
    errorElement.textContent = errorMessage;
    errorElement.classList.toggle("hidden", !errorMessage);
}

// This function prevents the same team from being selected in both dropdowns.
function syncTeamDropdownOptions() {
    const selectedBattingTeam = battingTeamSelect.value;
    const selectedBowlingTeam = bowlingTeamSelect.value;

    Array.from(battingTeamSelect.options).forEach((option) => {
        option.disabled = option.value === selectedBowlingTeam;
    });

    Array.from(bowlingTeamSelect.options).forEach((option) => {
        option.disabled = option.value === selectedBattingTeam;
    });
}

// This function checks that batting and bowling teams are different.
function validateDifferentTeamSelection() {
    const sameTeamSelected = battingTeamSelect.value === bowlingTeamSelect.value;
    const validationMessage = sameTeamSelected ? "Batting and bowling teams must be different." : "";

    battingTeamSelect.setCustomValidity(validationMessage);
    bowlingTeamSelect.setCustomValidity(validationMessage);
    updateErrorMessage(teamSelectionError, validationMessage);

    return !sameTeamSelected;
}

// This function updates the team labels shown above both innings forms.
function updateInningsTeamSummary() {
    firstInningsTeamSummary.textContent = `Batting: ${battingTeamSelect.value} | Bowling: ${bowlingTeamSelect.value}`;
    secondInningsTeamSummary.textContent = `Batting: ${bowlingTeamSelect.value} | Bowling: ${battingTeamSelect.value}`;
}

// This function parses a whole-number input and checks its valid range.
function parseWholeNumberFieldValue(value, fieldName, minimum, maximum = null) {
    const trimmedFieldValue = String(value).trim();
    if (!/^\d+$/.test(trimmedFieldValue)) {
        throw new Error(`${fieldName} must be a whole number.`);
    }

    const parsedNumber = Number.parseInt(trimmedFieldValue, 10);
    if (parsedNumber < minimum) {
        throw new Error(`${fieldName} must be at least ${minimum}.`);
    }
    if (maximum !== null && parsedNumber > maximum) {
        throw new Error(`${fieldName} cannot be more than ${maximum}.`);
    }
    return parsedNumber;
}

// This function validates overs input in cricket format like 10.3.
function parseOversFieldValue(value) {
    const oversFieldText = String(value).trim();
    if (!oversFieldText) {
        throw new Error("Overs completed is required.");
    }

    const oversParts = oversFieldText.split(".");
    if (oversParts.length > 2 || !/^\d+$/.test(oversParts[0]) || (oversParts[1] && !/^\d+$/.test(oversParts[1]))) {
        throw new Error("Overs must use cricket notation like 10.3.");
    }

    const completedOversCount = Number.parseInt(oversParts[0], 10);
    const ballsInCurrentOver = oversParts[1] ? Number.parseInt(oversParts[1], 10) : 0;

    if (completedOversCount < 0 || ballsInCurrentOver < 0) {
        throw new Error("Overs cannot be negative.");
    }
    if (ballsInCurrentOver > 5) {
        throw new Error("Balls in an over must be between 0 and 5.");
    }
    if (completedOversCount > 20 || (completedOversCount === 20 && ballsInCurrentOver > 0)) {
        throw new Error("Overs cannot exceed 20.");
    }

    return oversFieldText;
}

// This function returns the selected teams for the first-innings request.
function getFirstInningsTeamSelection() {
    if (!validateDifferentTeamSelection()) {
        throw new Error("Batting and bowling teams must be different.");
    }

    return {
        batting_team: battingTeamSelect.value,
        bowling_team: bowlingTeamSelect.value,
    };
}

// This function swaps the selected teams for the second-innings chase request.
function getSecondInningsTeamSelection() {
    if (!validateDifferentTeamSelection()) {
        throw new Error("Batting and bowling teams must be different.");
    }

    return {
        batting_team: bowlingTeamSelect.value,
        bowling_team: battingTeamSelect.value,
    };
}

// This function updates the button text and disabled state during requests.
function setSubmitButtonLoadingState(submitButtonElement, isLoading, defaultLabel) {
    submitButtonElement.disabled = isLoading;
    submitButtonElement.textContent = isLoading ? "Loading..." : defaultLabel;
}

// This function shows the first-innings prediction result in the UI.
function renderFirstInningsPrediction(predictionResult) {
    predictedScoreValueElement.textContent = predictionResult.predicted_score;

    if (predictionResult.innings_complete && predictionResult.target_score !== null) {
        targetScoreField.value = predictionResult.target_score;
        predictedScoreRangeElement.textContent = `Final score locked at ${predictionResult.predicted_score}. Target score updated to ${predictionResult.target_score}.`;
        return;
    }

    predictedScoreRangeElement.textContent = `Range: ${predictionResult.lower_bound}-${predictionResult.upper_bound} | Phase: ${predictionResult.phase}`;
}

// This function shows the second-innings win probabilities in the UI.
function renderSecondInningsPrediction(predictionResult) {
    battingTeamLabelElement.textContent = bowlingTeamSelect.value;
    bowlingTeamLabelElement.textContent = battingTeamSelect.value;
    battingTeamWinPercentageElement.textContent = `${predictionResult.batting_team_win_pct}%`;
    bowlingTeamWinPercentageElement.textContent = `${predictionResult.bowling_team_win_pct}%`;
    battingTeamWinBarElement.style.width = `${predictionResult.batting_team_win_pct}%`;
    bowlingTeamWinBarElement.style.width = `${predictionResult.bowling_team_win_pct}%`;
    winProbabilityMetaElement.textContent = `Required run rate: ${predictionResult.required_run_rate} | ${predictionResult.match_status}`;
}

// This handler refreshes the UI when the batting team selection changes.
battingTeamSelect.addEventListener("change", () => {
    updateInningsTeamSummary();
    syncTeamDropdownOptions();
    validateDifferentTeamSelection();
});

// This handler refreshes the UI when the bowling team selection changes.
bowlingTeamSelect.addEventListener("change", () => {
    updateInningsTeamSummary();
    syncTeamDropdownOptions();
    validateDifferentTeamSelection();
});

// This handler validates the first-innings form and requests a score prediction.
firstInningsForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    updateErrorMessage(firstInningsFormError, "");

    let requestPayload;
    try {
        requestPayload = {
            ...getFirstInningsTeamSelection(),
            current_score: parseWholeNumberFieldValue(
                document.getElementById("first_current_score").value,
                "Current score",
                0
            ),
            overs_completed: parseOversFieldValue(document.getElementById("first_overs_completed").value),
            wickets_fallen: parseWholeNumberFieldValue(
                document.getElementById("first_wickets_fallen").value,
                "Wickets fallen",
                0,
                10
            ),
        };
    } catch (validationError) {
        updateErrorMessage(firstInningsFormError, validationError.message);
        return;
    }

    const submitButtonElement = document.getElementById("first_submit");
    setSubmitButtonLoadingState(submitButtonElement, true, "Predict First Innings");

    try {
        const apiResponse = await fetch("/predict-first-innings", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify(requestPayload),
        });
        const predictionResult = await apiResponse.json();

        if (!apiResponse.ok) {
            throw new Error(predictionResult.error || "Unable to predict the first innings score.");
        }

        renderFirstInningsPrediction(predictionResult);
    } catch (requestError) {
        updateErrorMessage(firstInningsFormError, requestError.message);
    } finally {
        setSubmitButtonLoadingState(submitButtonElement, false, "Predict First Innings");
    }
});

// This handler validates the chase form and requests win probabilities.
secondInningsForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    updateErrorMessage(secondInningsFormError, "");

    let requestPayload;
    try {
        requestPayload = {
            ...getSecondInningsTeamSelection(),
            target_score: parseWholeNumberFieldValue(targetScoreField.value, "Target score", 1),
            current_score: parseWholeNumberFieldValue(
                document.getElementById("second_current_score").value,
                "Current score",
                0
            ),
            overs_completed: parseOversFieldValue(document.getElementById("second_overs_completed").value),
            wickets_fallen: parseWholeNumberFieldValue(
                document.getElementById("second_wickets_fallen").value,
                "Wickets fallen",
                0,
                10
            ),
        };
    } catch (validationError) {
        updateErrorMessage(secondInningsFormError, validationError.message);
        return;
    }

    const submitButtonElement = document.getElementById("second_submit");
    setSubmitButtonLoadingState(submitButtonElement, true, "Predict Win Probability");

    try {
        const apiResponse = await fetch("/predict-second-innings", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify(requestPayload),
        });
        const predictionResult = await apiResponse.json();

        if (!apiResponse.ok) {
            throw new Error(predictionResult.error || "Unable to predict second-innings win probability.");
        }

        renderSecondInningsPrediction(predictionResult);
    } catch (requestError) {
        updateErrorMessage(secondInningsFormError, requestError.message);
    } finally {
        setSubmitButtonLoadingState(submitButtonElement, false, "Predict Win Probability");
    }
});

syncTeamDropdownOptions();
validateDifferentTeamSelection();
updateInningsTeamSummary();
