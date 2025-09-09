"""
Generate 50,000 synthetic healthcare claims for model training.

The distributions here are calibrated to roughly match what I've seen in real
PBM claims data — lognormal billed amounts, ~25% denial rate, realistic
correlations between features and denial outcomes.
"""

import os
import random
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

RANDOM_SEED = 42
N_CLAIMS = int(os.getenv("N_SYNTHETIC_CLAIMS", "50000"))

DIAGNOSIS_CODES = [
    "E11.9",   # Type 2 diabetes, unspecified
    "E11.65",  # Type 2 diabetes with hyperglycemia
    "I10",     # Essential hypertension
    "I25.10",  # Atherosclerotic heart disease
    "I50.9",   # Heart failure, unspecified
    "J44.1",   # COPD with acute exacerbation
    "J44.9",   # COPD, unspecified
    "J06.9",   # Acute upper respiratory infection
    "M54.5",   # Low back pain
    "M17.11",  # Primary osteoarthritis, right knee
    "M79.3",   # Panniculitis, unspecified
    "N18.3",   # Chronic kidney disease, stage 3
    "N18.9",   # Chronic kidney disease, unspecified
    "G47.33",  # Obstructive sleep apnea
    "K21.0",   # GERD with esophagitis
    "F32.1",   # Major depressive disorder, moderate
    "F41.1",   # Generalized anxiety disorder
    "R10.9",   # Unspecified abdominal pain
    "Z23",     # Encounter for immunization
    "Z00.00",  # General adult medical exam
    "L40.0",   # Psoriasis vulgaris
    "D64.9",   # Anemia, unspecified
    "B20",     # HIV disease
    "C50.911", # Malignant neoplasm of breast
    "G43.909", # Migraine, unspecified
]

PROCEDURE_CODES = {
    "evaluation_management": ["99201", "99202", "99203", "99204", "99205",
                              "99211", "99212", "99213", "99214", "99215",
                              "99221", "99222", "99223", "99231", "99232"],
    "surgery": ["27447", "27130", "29881", "47562", "49505",
                "43239", "43249", "27446", "23472", "28296"],
    "radiology": ["71046", "71250", "72148", "73721", "74177",
                  "77067", "70553", "72141", "73221", "76700"],
    "pathology": ["80053", "85025", "80048", "83036", "82947",
                  "84443", "80061", "81001", "87086", "88305"],
    "medicine": ["90834", "90837", "90846", "96372", "96413",
                 "93000", "93306", "92014", "90460", "96365"],
}

ALL_CPT_CODES = [code for codes in PROCEDURE_CODES.values() for code in codes]

# These are commonly prescribed drugs — NDC codes are 11-digit
DRUG_NDC_CODES = [
    "00002323301",  # Insulin lispro
    "00074309460",  # Adalimumab (Humira)
    "00002140180",  # Duloxetine
    "59148004690",  # Amlodipine
    "00071015523",  # Atorvastatin
    "00093505698",  # Metformin
    "00781107710",  # Lisinopril
    "00591080101",  # Omeprazole
    "00378180510",  # Levothyroxine
    "00093317456",  # Sertraline
    "00591040101",  # Albuterol inhaler
    "00173071320",  # Fluticasone
    "68462039710",  # Gabapentin
    "51991081490",  # Hydrochlorothiazide
    "00093531501",  # Losartan
]

PROVIDER_SPECIALTIES = [
    "internal_medicine", "family_practice", "cardiology", "orthopedics",
    "gastroenterology", "dermatology", "psychiatry", "neurology",
    "oncology", "pulmonology", "endocrinology", "nephrology",
    "general_surgery", "emergency_medicine", "radiology",
    "anesthesiology", "ophthalmology", "urology", "pathology",
    "physical_therapy",
]

PLACES_OF_SERVICE = {
    "11": "Office",
    "21": "Inpatient Hospital",
    "22": "On Campus-Outpatient Hospital",
    "23": "Emergency Room",
    "31": "Skilled Nursing Facility",
    "81": "Independent Laboratory",
    "12": "Home",
    "20": "Urgent Care Facility",
    "24": "Ambulatory Surgical Center",
    "99": "Other",
}


def _generate_provider_pool(n_providers: int = 500) -> pd.DataFrame:
    """Create a pool of providers with NPIs and specialties."""
    rng = np.random.default_rng(RANDOM_SEED)
    npis = [f"{rng.integers(1000000000, 9999999999)}" for _ in range(n_providers)]
    specialties = rng.choice(PROVIDER_SPECIALTIES, size=n_providers)
    # Each provider has a baseline denial tendency (some are sloppier with paperwork)
    denial_tendency = rng.beta(2, 6, size=n_providers)
    return pd.DataFrame({
        "provider_npi": npis,
        "provider_specialty": specialties,
        "provider_denial_tendency": denial_tendency,
    })


def _generate_member_pool(n_members: int = 10000) -> pd.DataFrame:
    """Create a pool of members with demographics."""
    rng = np.random.default_rng(RANDOM_SEED + 1)
    ages = rng.normal(loc=52, scale=18, size=n_members).clip(18, 95).astype(int)
    genders = rng.choice(["M", "F"], size=n_members, p=[0.48, 0.52])
    return pd.DataFrame({
        "member_id": [f"MBR-{i:06d}" for i in range(n_members)],
        "member_age": ages,
        "member_gender": genders,
    })


def generate_claims(n_claims: int = N_CLAIMS) -> pd.DataFrame:
    """
    Generate synthetic healthcare claims with realistic distributions.

    The denial logic is intentionally correlated with features so the ML model
    has real signal to learn from. In production, the actual denial reasons are
    more complex (medical necessity reviews, prior auth requirements, etc.),
    but these synthetic correlations capture the main patterns.
    """
    rng = np.random.default_rng(RANDOM_SEED)
    random.seed(RANDOM_SEED)

    providers_df = _generate_provider_pool()
    members_df = _generate_member_pool()

    base_date = datetime(2024, 1, 1)
    records = []

    for i in range(n_claims):
        claim_id = f"CLM-{2024 + i // 25000}-{i:05d}"
        claim_type = rng.choice(["medical", "pharmacy", "dental"], p=[0.60, 0.30, 0.10])

        provider_row = providers_df.iloc[rng.integers(0, len(providers_df))]
        member_row = members_df.iloc[rng.integers(0, len(members_df))]

        service_date = base_date + timedelta(days=int(rng.integers(0, 730)))
        submission_lag = int(rng.exponential(scale=15)) + 1
        submission_date = service_date + timedelta(days=submission_lag)

        pos_code = rng.choice(list(PLACES_OF_SERVICE.keys()),
                              p=[0.40, 0.08, 0.15, 0.07, 0.03, 0.05, 0.04, 0.08, 0.05, 0.05])

        n_dx = rng.integers(1, 4)
        diagnosis_codes = list(rng.choice(DIAGNOSIS_CODES, size=n_dx, replace=False))

        if claim_type == "medical":
            cpt_category = rng.choice(list(PROCEDURE_CODES.keys()),
                                      p=[0.45, 0.10, 0.15, 0.20, 0.10])
            procedure_codes = list(rng.choice(PROCEDURE_CODES[cpt_category],
                                              size=rng.integers(1, 3), replace=False))
        elif claim_type == "dental":
            procedure_codes = [f"D{rng.integers(1000, 9999)}"]
        else:
            procedure_codes = []

        is_surgery = any(c in PROCEDURE_CODES.get("surgery", []) for c in procedure_codes)
        is_emergency = pos_code == "23"
        is_inpatient = pos_code == "21"

        if claim_type == "pharmacy":
            drug_ndc = str(rng.choice(DRUG_NDC_CODES))
            days_supply = int(rng.choice([30, 60, 90], p=[0.60, 0.25, 0.15]))
            billed_amount = float(rng.lognormal(mean=4.5, sigma=1.2))
        else:
            drug_ndc = None
            days_supply = None
            if is_surgery:
                billed_amount = float(rng.lognormal(mean=9.5, sigma=0.8))
            elif is_inpatient:
                billed_amount = float(rng.lognormal(mean=8.5, sigma=1.0))
            elif is_emergency:
                billed_amount = float(rng.lognormal(mean=7.0, sigma=0.9))
            else:
                billed_amount = float(rng.lognormal(mean=5.0, sigma=1.0))

        billed_amount = round(max(billed_amount, 5.0), 2)

        # Allowed amount is typically 40-85% of billed for in-network
        allowed_ratio = rng.uniform(0.40, 0.85)
        allowed_amount = round(billed_amount * allowed_ratio, 2)

        # Denial probability — this is where the signal lives
        denial_prob = 0.10  # baseline

        # Late submissions get denied more
        if submission_lag > 90:
            denial_prob += 0.35
        elif submission_lag > 60:
            denial_prob += 0.15
        elif submission_lag > 30:
            denial_prob += 0.05

        # High-cost claims face more scrutiny
        if billed_amount > 10000:
            denial_prob += 0.15
        elif billed_amount > 5000:
            denial_prob += 0.08

        # Provider tendency matters a lot
        denial_prob += provider_row["provider_denial_tendency"] * 0.3

        # Chronic conditions need better documentation, more denials
        chronic_prefixes = ["E11", "I10", "J44", "N18"]
        has_chronic = any(
            any(dx.startswith(p) for p in chronic_prefixes)
            for dx in diagnosis_codes
        )
        if has_chronic:
            denial_prob += 0.05

        # Emergency claims actually have lower denial rates
        if is_emergency:
            denial_prob -= 0.10

        # Pharmacy claims — formulary issues
        if claim_type == "pharmacy":
            denial_prob += 0.03

        # Inpatient — needs prior auth, higher denial if missing
        if is_inpatient:
            denial_prob += 0.08

        # Surgery — high scrutiny
        if is_surgery:
            denial_prob += 0.10

        denial_prob = np.clip(denial_prob, 0.02, 0.95)
        is_denied = int(rng.random() < denial_prob)

        records.append({
            "claim_id": claim_id,
            "member_id": member_row["member_id"],
            "provider_npi": provider_row["provider_npi"],
            "claim_type": claim_type,
            "place_of_service": pos_code,
            "diagnosis_codes": "|".join(diagnosis_codes),
            "procedure_codes": "|".join(procedure_codes) if procedure_codes else "",
            "billed_amount": billed_amount,
            "allowed_amount": allowed_amount,
            "service_date": service_date.strftime("%Y-%m-%d"),
            "submission_date": submission_date.strftime("%Y-%m-%d"),
            "drug_ndc": drug_ndc or "",
            "days_supply": days_supply if days_supply else 0,
            "member_age": member_row["member_age"],
            "member_gender": member_row["member_gender"],
            "provider_specialty": provider_row["provider_specialty"],
            "is_denied": is_denied,
        })

    claim_df = pd.DataFrame(records)

    actual_denial_rate = claim_df["is_denied"].mean()
    print(f"Generated {len(claim_df):,} claims")
    print(f"Denial rate: {actual_denial_rate:.1%}")
    print(f"Claim types: {claim_df['claim_type'].value_counts().to_dict()}")
    print(f"Billed amount: mean=${claim_df['billed_amount'].mean():,.2f}, "
          f"median=${claim_df['billed_amount'].median():,.2f}")

    return claim_df


def main():
    claim_df = generate_claims()

    output_dir = Path(os.getenv("DATA_DIR", "data"))
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "synthetic_claims.csv"

    claim_df.to_csv(output_path, index=False)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
