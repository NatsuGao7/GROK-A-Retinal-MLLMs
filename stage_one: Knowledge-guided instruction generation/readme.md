![Knowledge_instruction](./Knowledge_instruction.png)
# stage_one: Knowledge-guided instruction generation
## Data preprocessing (Geneartion diagnosis label from UKBiobank)

### Hypertension
***1.*** **Self-reported doctor‐diagnosed hypertension (Field ***6150***):**  
– Include only participants who reported hypertension and no other vascular/heart conditions; exclude anyone who reported multiple conditions including hypertension.  
– Label those who selected “None of the above” as normotensive.  
– After this filtering, ***121,033*** individuals are classified as hypertensive and ***350,662*** as normal.

***2.*** **Blood pressure measurements (systolic and diastolic):**  
– For each participant, compute the average of automated and manual readings of systolic blood pressure (SBP), and similarly for diastolic blood pressure (DBP).  
– Classify anyone with average SBP ≥ 140 mmHg or average DBP ≥ 90 mmHg as hypertensive; otherwise, label as normal.  
– Merge these measurement‐based labels with the self‐report results from Step ***1***. After combining, there are ***252,047*** hypertensive and ***219,648*** normal individuals.

***3.*** **Medication records (Fields ***6153*** and ***6177***):**  
– Identify participants who take only “blood pressure medication” (and no medications for cholesterol, diabetes, etc.) and label them as hypertensive.  
– Label those who selected “None of the above” as normal.

***4.*** **Final merging:**  
– Combine the medication‐based labels from Step 3 with the combined self‐report and measurement labels from Step 2.  
– In the end, ***202,005*** individuals are classified as hypertensive and ***204,996*** as normal. 
### Age-related Macular Degeneration (ADM)
1) **ICD-10 main diagnosis** (UKB Field **41202**)  
   - If any column equals **`H353`** (age-related macular degeneration), set **amd = 1**; otherwise **amd = 0**.

2) **Which eye(s) affected by macular degeneration** (Field **5912**)  
   - This field records laterality: typically **1 = left**, **2 = right**, **3 = both**.  
   - If counting “any affected” as positive: values **∈ {1, 2, 3}** are positive.  
   - If counting “both eyes only”: value **= 3** is positive.

3) **Age macular degeneration diagnosed** (Field **5923**)  
   - If the value is **> 0** (a diagnosis age exists), mark as positive.

4) **Eye problems/disorders** (Field **6148**)  
   - If any instance equals **5** (corresponding to AMD), mark as positive.

5) **Non-cancer illness code, self-reported** (Field **20002**)  
   - If any instance equals **1528** (macular degeneration), mark as positive.
### Glaucoma
