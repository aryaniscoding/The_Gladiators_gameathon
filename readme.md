
---

###  Downloadable `README.md`

I’ve prepared the README file content for easy download. Click the button below to download it directly and add it to your repo:

<button onclick="downloadREADME()">Download README.md</button>

<script>
function downloadREADME() {
  const content = `# 🏏 The Gladiators: Fantasy Cricket Team Selector

## 🔍 Overview

**The Gladiators** is a machine learning-based application designed to automatically select the optimal fantasy cricket team for Indian T20 League matches. It analyzes historical player statistics and live squad data to recommend a balanced playing 11 and 4 backup players.

The selection process ensures team balance across roles (Wicket Keeper, Batsman, Bowler, All-Rounder) while optimizing for performance.

---

## 📁 Project Structure

\`\`\`
The_Gladiators_gameathon/
├── run_model.py               # Main script to run team selection
├── ipl_2025_gameathon.xlsx    # Player statistics and squad data
├── Dockerfile                 # Docker container setup
├── requirements.txt           # Python package dependencies
└── README.md                  # This documentation file
\`\`\`

---

## ⚙️ How It Works

### 📚 Data Inputs

1. **Squad Excel File**: Contains match-wise list of available players.  
2. **Player Stats**: Batting average, strike rate, consistency, form, etc.

### 🚀 Selection Process

1. **Load Data**: Reads squad and player stats from the Excel file.  
2. **Feature Engineering**:
   - Calculates performance index = \`bat_avg × strike_rate / 100\`
   - Adds features like experience (log-transformed) and recent form.  
3. **Model Training**:
   - Uses \`RandomForestRegressor\` with cross-validation and hyperparameter tuning.  
4. **Team Selection**:
   - Ensures at least one player from each type (WK, BAT, BOWL, ALL)  
   - Picks top performers  
   - Chooses a Captain (C) and Vice-Captain (VC) from different teams  
   - Selects 4 backup players  
5. **Output**:
   - A \`.csv\` file is generated at \`~/Downloads/The_Gladiators_output.csv\`

---

## 📊 Machine Learning Model

- **Model**: Random Forest Regressor  
- **Key Features**:
  - Batting average & strike rate  
  - Match experience  
  - Recent performance (last 3 games)  
  - Player type & team (categorical)  
- **Libraries**: \`pandas\`, \`numpy\`, \`scikit-learn\`

---

## 🐳 Docker Support

A Dockerfile is included to containerize the entire project for consistent environments.

---

## ⚠️ Important Notes

- **Before Every Match**: Delete the previous \`The_Gladiators_output.csv\` from your Downloads folder.  
- **Ensure Proper Sheet Naming**: The Excel sheet should be named \`Match_XX\` where \`XX\` is the match number.  
- **Player Name Matching**: Ensure names in the squad and stats match exactly.  
- **Output Location**: File is saved to your Downloads directory.

---

## 🚀 How to Run This Project

###  Option 1: Run Locally (Recommended)

\`\`\`bash
git clone https://github.com/aryaniscoding/The_Gladiators_gameathon.git
cd The_Gladiators_gameathon
\`\`\`

Set up and activate a virtual environment:

\`\`\`bash
python -m venv .venv
# On Linux/macOS:
source .venv/bin/activate
# On Windows:
.\\.venv\\Scripts\\activate
\`\`\`

Install dependencies and run:

\`\`\`bash
pip install -r requirements.txt
python run_model.py
\`\`\`

---

###  Option 2: Run Using Docker

\`\`\`bash
docker build -t gladiators-selector .
docker run -v ~/Downloads:/app/Downloads gladiators-selector
\`\`\`

Ensure that \`ipl_2025_gameathon.xlsx\` is included or mounted properly.

---

## 🧠 Contributors

Made with passion for Gameathon 2025.

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).
`;
  const blob = new Blob([content], { type: 'text/markdown' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = "README.md";
  a.click();
  URL.revokeObjectURL(url);
}
</script>
