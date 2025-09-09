"""
CPT code to clinical category mapping.

This covers the most common codes I see in claims data. The full CPT codebook
has 10,000+ codes, but these 60+ cover probably 80% of the volume we process.
"""

CPT_TO_CATEGORY: dict[str, str] = {
    # Evaluation & Management — office visits, consultations
    "99201": "evaluation_management",
    "99202": "evaluation_management",
    "99203": "evaluation_management",
    "99204": "evaluation_management",
    "99205": "evaluation_management",
    "99211": "evaluation_management",
    "99212": "evaluation_management",
    "99213": "evaluation_management",  # most common outpatient code by far
    "99214": "evaluation_management",
    "99215": "evaluation_management",
    "99221": "evaluation_management",  # initial hospital care
    "99222": "evaluation_management",
    "99223": "evaluation_management",
    "99231": "evaluation_management",  # subsequent hospital care
    "99232": "evaluation_management",
    "99233": "evaluation_management",
    "99281": "evaluation_management",  # ED visit level 1
    "99282": "evaluation_management",
    "99283": "evaluation_management",
    "99284": "evaluation_management",
    "99285": "evaluation_management",  # ED visit level 5 (highest)
    "99354": "evaluation_management",  # prolonged services
    "99355": "evaluation_management",

    # Surgery — orthopedic, general, GI
    "27447": "surgery",  # total knee replacement
    "27130": "surgery",  # total hip arthroplasty
    "29881": "surgery",  # knee arthroscopy with meniscectomy
    "47562": "surgery",  # laparoscopic cholecystectomy
    "49505": "surgery",  # inguinal hernia repair
    "43239": "surgery",  # upper GI endoscopy with biopsy
    "43249": "surgery",  # esophagogastroduodenoscopy with dilation
    "27446": "surgery",  # revision knee arthroplasty
    "23472": "surgery",  # total shoulder arthroplasty
    "28296": "surgery",  # bunionectomy
    "44970": "surgery",  # laparoscopic appendectomy
    "58661": "surgery",  # laparoscopic excision of ovarian cyst

    # Radiology — imaging, CT, MRI, X-ray
    "71046": "radiology",  # chest X-ray 2 views
    "71250": "radiology",  # CT chest without contrast
    "72148": "radiology",  # MRI lumbar spine without contrast
    "73721": "radiology",  # MRI lower extremity joint
    "74177": "radiology",  # CT abdomen/pelvis with contrast
    "77067": "radiology",  # screening mammography
    "70553": "radiology",  # MRI brain with and without contrast
    "72141": "radiology",  # MRI cervical spine without contrast
    "73221": "radiology",  # MRI upper extremity joint
    "76700": "radiology",  # abdominal ultrasound
    "76856": "radiology",  # pelvic ultrasound
    "73610": "radiology",  # ankle X-ray

    # Pathology & Laboratory
    "80053": "pathology",  # comprehensive metabolic panel
    "85025": "pathology",  # complete blood count with differential
    "80048": "pathology",  # basic metabolic panel
    "83036": "pathology",  # hemoglobin A1c
    "82947": "pathology",  # glucose, quantitative
    "84443": "pathology",  # thyroid stimulating hormone (TSH)
    "80061": "pathology",  # lipid panel
    "81001": "pathology",  # urinalysis with microscopy
    "87086": "pathology",  # urine culture
    "88305": "pathology",  # surgical pathology, gross and micro
    "86900": "pathology",  # blood typing ABO
    "82306": "pathology",  # vitamin D

    # Medicine — therapeutic, psychiatric, infusions
    "90834": "medicine",  # psychotherapy 45 min
    "90837": "medicine",  # psychotherapy 60 min
    "90846": "medicine",  # family psychotherapy without patient
    "96372": "medicine",  # therapeutic injection
    "96413": "medicine",  # chemotherapy infusion first hour
    "93000": "medicine",  # electrocardiogram (ECG)
    "93306": "medicine",  # echocardiography
    "92014": "medicine",  # ophthalmological exam
    "90460": "medicine",  # immunization admin first component
    "96365": "medicine",  # IV infusion first hour
    "97110": "medicine",  # therapeutic exercises (PT)
    "97140": "medicine",  # manual therapy techniques
}

CATEGORY_DESCRIPTIONS = {
    "evaluation_management": "Office visits, hospital visits, consultations, and ED encounters",
    "surgery": "Surgical procedures including orthopedic, GI, and general surgery",
    "radiology": "Imaging services: X-ray, CT, MRI, ultrasound, mammography",
    "pathology": "Laboratory tests and surgical pathology",
    "medicine": "Therapeutic services, psychiatric, infusions, physical therapy",
}

# High-cost procedure codes that tend to get more scrutiny from payers
HIGH_COST_PROCEDURES = {
    "27447", "27130", "23472", "27446",  # joint replacements
    "96413",  # chemo infusion
    "70553", "74177",  # advanced imaging
    "44970", "47562", "58661",  # laparoscopic surgeries
}

# TODO: add ASC (Ambulatory Surgery Center) fee schedule codes — they have
# different reimbursement rates and I want to capture that in the model


def get_cpt_category(cpt_code: str) -> str:
    """Map a CPT code to its clinical category. Returns 'other' for unknown codes."""
    return CPT_TO_CATEGORY.get(cpt_code, "other")


def is_high_cost_procedure(cpt_code: str) -> bool:
    return cpt_code in HIGH_COST_PROCEDURES


def get_procedure_complexity(cpt_codes: list[str]) -> dict[str, int]:
    """Count procedures by category for a claim. Useful as features."""
    category_counts: dict[str, int] = {}
    for code in cpt_codes:
        cat = get_cpt_category(code)
        category_counts[cat] = category_counts.get(cat, 0) + 1
    return category_counts
