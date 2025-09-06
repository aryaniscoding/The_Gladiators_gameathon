# The Gladiators Fantasy Cricket Team Selector

## Overview
This application uses machine learning to automatically select an optimal fantasy cricket team for Indian T20 League matches. The system analyzes player statistics and match data to identify the best possible playing 11 along with 4 backup players, while ensuring team balance with different player types (Wicket Keeper, Batsman, Bowler, All-Rounder).

## Project Structure
```
The_Gladiators_gameathon/
├── run_model.py          # Main Python script that selects the team
├── ipl_2025_gameathon.xlsx  # Player statistics database
├── Dockerfile            # Container configuration
├── requirements.txt      # Python dependencies
└── README.md            # This documentation file
```

## How It Works

### Data Sources
The application uses two key data sources:
1. **Squad Information**: An Excel file containing the players available for each match
2. **Player Statistics**: Historical performance data including batting average, strike rate, consistency, and recent form

### Selection Process
1. **Data Loading**: The script loads the playing squad for the specified match and merges it with player statistics
2. **Feature Engineering**: Creates derived features like experience level and performance index
3. **Model Training**: Trains a Random Forest Regressor using cross-validation to predict player performance
4. **Player Selection**: 
   - Selects players based on predicted scores
   - Ensures at least one player of each required type (WK, BAT, BOWL, ALL)
   - Selects the remaining spots based on predicted performance
   - Designates captain (C) and vice-captain (VC), preferably from different teams
   - Selects 4 backup players
5. **Output Generation**: Creates a CSV file with the selected team in The_Gladiators format

### Machine Learning Approach
The system uses a Random Forest Regressor with hyperparameter tuning to predict player performance. Features include:
- Batting average and strike rate
- Match experience (transformed with log)
- Performance index (calculated as avg × sr / 100)
- Recent form (points from last 3 matches)
- Team and player type as categorical variables

## Running the Application

### Prerequisites
- Docker installed on your system
- Excel files with squad and statistics data


### Important Notes

⚠️ **PLEASE NOTE**: 
- The model training process takes time to complete. Please be patient after running the command - it may take half a minute to generate results depending on your system's specifications.

- **BEFORE EACH NEW MATCH**: Delete the previous `The_Gladiators_output.csv` file from your Downloads folder. If you don't delete the previous output file, the script may not update it correctly due to file permission constraints.

## Output Format
The script generates a CSV file at `/Downloads/The_Gladiators_output.csv` with the following columns:
- **Player name**: Player's full name
- **Team**: Player's team in the league
- **C/VC**: Captain (C), Vice-Captain (VC), or NA designation

The output contains:
- 11 selected players (first 11 rows)
- A blank row separator
- 4 backup players (last 4 rows)

## Technical Details
- **Language**: Python 3.10
- **Key Libraries**: pandas, numpy, scikit-learn
- **Model**: RandomForestRegressor with hyperparameter optimization
- **Containerization**: Docker for easy deployment and execution

## Troubleshooting
- If the output file is not generated, check file permissions in your Downloads folder
- Ensure the squad Excel file has the correct sheet name format: "Match_XX"
- Verify that player names match exactly between the squad and statistics files