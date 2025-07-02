# LLM-Rationality-Benchmark


# Overview of files
## ./SM_measurement_questions.docx
The measured questions in the benchmark, including all the scales and tests in **psychology**, **cognitive and behaviorial science**, **decision-making**, **economics**, and **societal** domains, and prompts for **game theory** and **cooperation and coordination** domains.  
(These questions are also included in the code file.)

## ./code_release 
Code for the benchmark and raw results in the paper.
### code
- **Psychology&Cognitive&DecisionMaking&Economics.ipynb**  
Code for Psychology, Cognitive and Behavioral Sciences, Decision-Making Theory and Economics domains.

- **Game_theory&cooperation_coordination.ipynb**  
Code for game theory and societal (cooperation and coordination) domains.

- **Wisdom_of_crowds.ipynb**  
Code for societal (wisdom of crowds) domains.

- **Analysis_Survey.ipynb**  
Calculate rationality for Psychology, Cognitive and Behavioral Sciences, Decision-Making Theory and Economics domains.

- **Plot_Survey.ipynb**, **Plot_game_social.ipynb**, **Plot_domain.ipynb**  
Plot result figures.

### results
- **survey_result.xlsx**  
The raw answer of LLMs to questions in Psychology, Cognitive and Behavioral Sciences, Decision-Making Theory and Economics domains.

- **survey_analysis.xlsx**  
The rationality score (before normalization) of LLMs in Psychology, Cognitive and Behavioral Sciences, Decision-Making Theory and Economics domains.

- **game_results.xlsx**  
The rationality score (before normalization) of LLMs in game theory and societal (cooperation and coordination) domains.

- **domain_results.xlsx**
The overall rationality score of LLMs in each domain.


# Usage of toolkit
## Step 1: Run experiments.

Set up LLM configs in LLM_setup.py, then run
```
python run.py
```

You can also run aspects one by one in jupyter notebook as follows:

Rationality test for  **psychology**, **cognitive and behaviorial science**, **decision-making**, **economics** domains:   
```
Run the scripts in Psychology&Cognitive&DecisionMaking&Economics.ipynb
```
Rationality test for **game theory**, **cooperation and coordination**, and **wisdom of crowds** domains:
```
Run the scripts in Game_theory&cooperation_coordination.ipynb and Wisdom_of_crowds.ipynb
```
The detailed instructions about LLM setup and experiments are included in the jupyter notebook files.

## Step 2: Result analysis
Run the scripts in **Analysis_Survey.ipynb** to generate rationality scores.

## Step 3: Result visualization through heatmap
Run the scripts in **Plot_Survey.ipynb**, **Plot_game_social.ipynb**, and **Plot_domain.ipynb**.

