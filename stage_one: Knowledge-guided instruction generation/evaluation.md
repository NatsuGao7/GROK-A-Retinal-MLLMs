# Task
Evaluate how well the *Generated Retinal/OCT Interpretation* matches a *Ground-Truth Clinician Report*, given the original images and quantitative biomarkers as references.

## Inputs
- **GroundTruthReport:** {{groundtruth_report}}
- **BiomarkersJSON (optional):** {{biomarkers_json}} 
- **GeneratedReports:**  
  1. **ModelName:** "model_1"
     **ReportText:** {{model_1}}  
       

## Evaluation Criteria & Scoring

### 1. DiagnosisAccuracy (0–20)
**What:** Is the primary diagnosis (and stage/laterality/severity) correct and specific?  
**How to score:**  
- 20 = Exact match (diagnosis + stage/grade/side).  
- 15 = Diagnosis right, minor mismatch in stage/severity/laterality.  
- 10 = Correct disease family but key attribute wrong (e.g., “advanced” vs “mild”).  
- 5  = Partially correct elements, overall wrong.  
- 0  = Completely wrong.

### 2. QuantitativeAccuracy (0–20)
**What:** Are numeric statements (e.g., CDR, GC-IPL, RNFL, central thickness, AVR) correct?  
**How to score (start at 20, subtract):**  
- −0: Value correct (or within ±10%) and tied to finding.  
- −1: Value off by >10% but ≤20%, or unit mismatch.  
- −2: Off by >20% or clearly wrong magnitude.  
- −3: Fabricated number or contradicts provided biomarker. 
-  0：Analysis without any quantitative statement should be directly punished to 0.
(Floor = 0)

### 3. QualitativeAccuracy (0–15)
**What:** Are descriptive findings (disc swelling, rim color, hemorrhages, drusen, CWS, edema, etc.) accurate?  
**How to score:**  
- 15 = All major qualitative findings correct; no fabricated lesions.  
- 10 = Mostly correct; minor omissions or mild mischaracterizations.  
- 5  = Several key features missing or inverted.  
- 0  = Largely incorrect / invented features.

### 4. EvidenceGrounding (0–10)
**What:** Does the report explicitly ground claims in data (numbers, visible signs on images)?  
**How to score:**  
- 10 = Consistently cites metrics/findings as evidence for each conclusion.  
- 7  = Frequent evidence use but not systematic.  
- 3  = Sporadic evidence references.  
- 0  = No explicit evidence linkage.

### 5. ReasoningConsistency (0–15)
**What:** Is the logic from findings → inference → diagnosis coherent and non-contradictory?  
**How to score:**  
- 15 = Clear, stepwise reasoning; no leaps.  
- 10 = Generally coherent; 1–2 small jumps or weak links.  
- 5  = Multiple gaps or cause–effect reversals.  
- 0  = Incoherent or self-contradictory.

### 6. CoverageCompleteness (0–10)
**What:** Does the interpretation address the key anatomical/functional domains?  
Domains to check (score +2 each if adequately covered and mostly correct):  
1) Optic disc/optic nerve  
2) Macula/fovea  
3) Retinal vasculature  
4) Peripheral/other retinal findings  
5) OCT layer analysis (GC-IPL, RNFL, ISOS–RPE, etc.)  
(Max = 10; if a domain is described but clearly wrong, do not award its points.)

### 7. ImageBiomarkerAlignment (0–10)
**What:** Do textual claims align with the actual images/biomarkers provided?  
**How to score:**  
- 10 = Strong alignment; no contradictions.  
- 7  = Minor mismatches.  
- 3  = Several inconsistencies.  
- 0  = Mostly misaligned or ignores provided data.

### 8. ErrorSeverityPenalty (−15–0)
**What:** Penalize dangerous or management-changing errors.  
**How to score:**  
- −15 = Major misdiagnosis likely to alter treatment (e.g., calling normal eye “advanced glaucoma”).  
- −10 = Severe over/under-calling critical pathology.  
- −5  = Moderate but notable error.  
- 0   = No serious error.

---

## Output Format
Return **only** the JSON object below (no extra text) for each model, the explanation should be concise. Note that "Diognosised Disease" means the diagnosis from the report, not the groundtruth answer, this is used for caculation of F1 score:

{ "model_1":{
  "DiagnosisAccuracy":      { "Score": 0,  "Explanation": "" },
  "QuantitativeAccuracy":   { "Score": 0,  "Explanation": "" },
  "QualitativeAccuracy":    { "Score": 0,  "Explanation": "" },
  "EvidenceGrounding":      { "Score": 0,  "Explanation": "" },
  "ReasoningConsistency":   { "Score": 0,  "Explanation": "" },
  "CoverageCompleteness":   { "Score": 0,  "Explanation": "" },
  "ImageBiomarkerAlignment":{ "Score": 0,  "Explanation": "" },
  "ErrorSeverityPenalty":   { "Score": 0,  "Explanation": "" },
  "TotalScore": 0，
  "Diognosised Disease: (Choose one from Alzheimer, Glaucoma, Diabetes, DiabeticRetinopathy, Hypertension, AMD and Normal)
}}
