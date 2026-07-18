// Sync slider display values instantly
const inputs = ['Anxiety_Level_During_Exams', 'Perceived_AI_Dependency', 'Skill_Retention_Score'];
const labels = ['anxietyVal', 'depVal', 'retentionVal'];

inputs.forEach((id, idx) => {
    document.getElementById(id).addEventListener('input', (e) => {
        document.getElementById(labels[idx]).textContent = e.target.value;
    });
});

document.getElementById('metricsForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    // Exact mapping tracking all 13 ML features
    const fields = [
        'Major_Category', 'Year_of_Study', 'Pre_Semester_GPA', 'Weekly_GenAI_Hours',
        'Primary_Use_Case', 'Prompt_Engineering_Skill', 'Tool_Diversity', 
        'Paid_Subscription', 'Traditional_Study_Hours', 'Perceived_AI_Dependency',
        'Institutional_Policy', 'Anxiety_Level_During_Exams', 'Skill_Retention_Score'
    ];
    
    const payload = {};
    fields.forEach(f => {
        payload[f] = document.getElementById(f).value;
    });

    try {
        // Pointing directly to your live running Flask pipeline instance
        const res = await fetch('http://127.0.0.1:5000/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        const data = await res.json();
        
        if (data.status === 'success') {
            const consoleBox = document.getElementById('outputConsole');
            const rLabel = document.getElementById('riskLabel');
            consoleBox.classList.remove('hidden');
            
            rLabel.textContent = data.burnout_risk + " Risk Level";
            document.getElementById('confidenceLabel').textContent = data.confidence_score;
            
            // Apply dynamic theme colors based on model calculations
            if (data.burnout_risk === 'High') rLabel.style.color = '#ff3131';
            else if (data.burnout_risk === 'Medium') rLabel.style.color = '#ffb300';
            else rLabel.style.color = '#39ff14';
        } else {
            alert("Backend processing exception: " + data.message);
        }
    } catch (err) {
        console.error("Network Error: ", err);
        alert("Could not communicate with machine learning backend server. Ensure app.py is running!");
    }
});