# Your Task: Interpret the provided fundus photo (CFP), a representative OCT B-scan, and a list of detailed quantitative biomarkers. Your task is to generate a clinical assessment supported by observed evidence and provided information.

## Key Objectives:
1.  **Simulate a Realistic Ophthalmic Diagnostic Process**: Reflect how an ophthalmologist would analyze multimodal retinal images, correlate findings from different sources, and form a diagnostic impression.
2. **Grounded Retinal Understanding**: The analysis must be grounded in evidence. This means either linking clear visual signs to their corresponding quantitative biomarkers or drawing conclusions directly from the provided quantitative data when visual evidence is limited.
3.  **Evidence-Based Reasoning**: Support your diagnostic conclusion with a clear, logical reasoning process that synthesizes all available evidence into a cohesive narrative.


## Guidelines for the Retinal Image Analysis:
### 1. Provided Data:
**Crucial Context**: Subjects are aged between 40-69 years. All OCT measurements, including RNFL and GC-IPL thickness, are derived from a 6x6mm macular cube scan, not a peripapillary (optic disc) scan. Therefore, standard peripapillary RNFL diagnostic criteria do not apply.
*   **CFP Image**: A color fundus photograph showing the posterior pole of the eye, including the optic disc, macula, and major retinal vessels.
*   **OCT B-scan Image**: A cross-sectional OCT scan showing the retinal layers, in most cases, it passes through the fovea.
*   **Quantitative Biomarkers**: A comprehensive list of measurements from 3D OCT scan and CFP. 
*   **Final Diagnosis Report**: The established clinical diagnosis for this case, which your evidence-based analysis should logically support.

### 2. Role-play as an Ophthalmologist:
Act as a retina specialist. Analyze the provided images and biomarkers step-by-step:
*  **Analyze CFP**: Examine the fundus photo for optic disc health (cup, rim, color...), vessel condition (shape, A/V ratio, tortuosity, fractal dimension...), and gross pathologies (hemorrhages, exudates, drusen). Use `CFP_Measures` to ground your initial findings, but state only the cup-to-disc ratio numerically if needed; describe vessel tortuosity and fractal complexity qualitatively, since no widely accepted numeric standards exist.
*   **Examine OCT B-scan**: 
    1. Use the provided representative B-scan to understand the foveal contour and central retinal layer status. Correlate this with the central subfield measures from your quantitative biomarkers. **BUT do not reveal that you are provided with only a single B-scan**. 
    2. Act as if you have reviewed the full 3D OCT scan. using the comprehensive `Quantitative Biomarkers` to inform your analysis of the entire macular cube. For example, use the ETDRS grid data to describe spatial patterns of edema or atrophy beyond the central B-scan.
*   **Formulate a Cohesive Narrative**: Weave your findings from all sources into a single, evidence-based diagnostic story. And the process should be grounded.
    *   **Good Example**: "The OCT scan reveals significant thinning of the retinal nerve fiber layer, **the average RNFL thickness seems to be low, approximately 22um**, which strongly suggests glaucomatous damage."
    *   **Bad Example**: "The RNFL thickness is 22um." (Lacks visual context), or "The OCT scan suggests glaucomatous damage." (Lacks grounding).


### 3. Handling Quantitative Biomarkers:
*   Treat the biomarkers as if they are derived or measured from the 3D OCT scan or CFP by yourself, rather than a given biomarkers list. Do not use phrases like "the provided measurements show...". And also try to avoid language that sounds like you are referencing an external data sheet, such as "this corresponds to...", "the value for this is..." or "the quantitative report confirms...".
*   **Instruction for Grounding**: Instead of just stating a number, use it to support a qualitative observation you've made from the images. You can choose appropriate case to use phrases like: "estimated to be around...", "likely in the range of...", "suggesting a value of approximately...", "corresponds to a thickness of about...".


### 4. Handling Missing Data: 
*   When a biomarker is missing (marked as null, N/A), if it is necessary for your diagnosis, **you must not omit its discussion**. You can rely on strong visual evidence and the Final Diagnosis Report to **provide a reasonable, semi-quantitative estimate** for that biomarker.


## Guidelines for Response Generation:
1.  Synthesize all findings to deduce a likely diagnosis. Clearly explain how the combined evidence from CFP, OCT, and biomarkers supports this conclusion.
2.  Your final diagnosis must match the provided `Final Diagnosis Report`. Do not invent a new diseases or diagnoses. But the analizing and resoning process should be systematic and comprehensive: 
    * Avoid confirmatory bias – Do not focus solely on biomarkers that support `Final Diagnosis Report`, treat all provided data as equally relevant until weighed in the narrative. Do not cherry-pick data that merely support the report. Your reasoning should show a step-by-step evaluation and elimination of alternative diagnoses, not a reverse-engineered justification.
    * Since the report only gives general disease categories, you can provide more specific details based on your analysis. For example, if the report says "AMD", specify if your findings point towards dry or wet AMD. Meanwhile, each disease may be mild, at an early stage or severe. Notice that, people with diabetes may not necessarily have diabetic retinopathy, but patients with DR must have diabetes.
3.  The `Final Diagnosis Report` serves only as the ground truth reference for you. The generated text **MUST NOT show that you are aware of the existence of the report**.
    * Only mention conditions you have actively ruled out on the basis of clear imaging or quantitative evidence. Do not list an entire block of diseases that merely happen to be absent.
    * When you need to exclude something, do it one at a time and link each exclusion to the specific finding that supports it.
    * **Bad example**: "In the absence of diabetic, hypertensive, glaucomatous, or age-related macular pathology, the eye is considered normal."(Reads like a memorised answer set and hints at a hidden ground-truth list.)
    * **Good example**: "The macula shows no micro-aneurysms, hemorrhages, or lipid exudates, making diabetic macular disease unlikely, and the neuro-retinal rim remains full without focal notching, so glaucomatous damage is not suspected."(Only two conditions are ruled out, and each is tied to a concrete observation.)
4.  If the `Final Diagnosis Report` is "Normal", it means there are no findings consistent with the specific diseases of interest. It does not necessarily mean the eye is perfectly healthy. You should still describe any minor, non-specific findings (e.g., mild vessel tortuosity, small drusen not meeting AMD criteria). Your final assessment should conclude that there is "no evidence of significant retinal pathology" or a similar phrase.
5.  Strictly follow the output format and requirements specified in your task instructions.
6.  Never make up explanations.

## Final Diagnosis Report:
{{Final Diagnosis Report}} 

## Quantitative Biomarkers:
{{Quantitative Biomarkers}} 
(Reminder: All OCT thickness values, including RNFL and GC-IPL, are derived from macular scan and are not peripapillary values.)

## Present your work in this format:
<response>[A comprehensive, narrative-style clinical assessment, written in a single paragraph and do not use list. Limit your responses within 300 words, reflecting a concise clinical summary.]
</response>