# Home Loan Eligibility Policy

Internal underwriting policy for retail home loans. Applies to salaried and
self-employed resident individual applicants.

## 1. Basic Eligibility

- Applicant age: minimum 21 years at application, maximum 65 years at loan maturity.
- Minimum net annual income: ₹3,00,000 for salaried applicants, ₹4,00,000 for
  self-employed applicants (based on average of last 2 years' ITR).
- Applicant must be a resident individual with a minimum of 2 years' continuous
  employment or business vintage.
- Co-applicant income can be clubbed for eligibility if the co-applicant is an
  immediate family member and a co-owner of the property.

## 2. Credit Score Bands

Credit score (CIBIL) determines both eligibility and the interest rate applied.
See `interest_rate_card.csv` for the exact rate per band.

| Band | CIBIL Score | Eligibility | Notes |
|---|---|---|---|
| Prime | 750 - 900 | Eligible | Best available rate, fastest processing |
| Standard | 700 - 749 | Eligible | Standard rate |
| Sub-prime | 650 - 699 | Conditionally eligible | Higher rate, may require additional collateral or a co-applicant |
| Below policy | 300 - 649 | Not eligible | Application to be declined, applicant may reapply after 6 months |

## 3. Debt-to-Income (DTI) / FOIR Cap

The bank uses the Fixed Obligation to Income Ratio (FOIR), referred to here as
DTI for consistency: `DTI = (existing monthly EMI obligations + proposed EMI) /
net monthly income`.

- Maximum permissible DTI for salaried applicants: **50%**.
- Maximum permissible DTI for self-employed applicants: **45%** (higher income
  volatility).
- If the proposed loan pushes DTI above the cap, the maximum eligible loan
  amount must be recalculated downward so that DTI lands at or below the cap
  (by reducing the loan amount and/or extending tenure), rather than declining
  the application outright, provided the applicant is otherwise eligible.

## 4. Maximum Eligible Loan Amount

The maximum eligible loan amount is derived from the maximum EMI the applicant
can service within the DTI cap, then converting that EMI to a principal amount
at the applicable interest rate and requested tenure using the standard
reducing-balance EMI formula:

```
max_affordable_emi = (DTI_cap x net_monthly_income) - existing_monthly_emis
max_eligible_loan  = max_affordable_emi converted to principal at the
                      applicable rate and tenure
```

If `max_affordable_emi` is zero or negative, the applicant is not eligible for
any additional loan until existing obligations reduce.

## 5. Tenure

- Minimum tenure: 5 years.
- Maximum tenure: 30 years, capped so that the loan matures by the applicant's
  65th birthday, whichever is shorter.

## 6. Loan-to-Value (LTV) Ratio

LTV is capped by loan amount slab per RBI guidelines (see `rbi_guidelines.md`).
The lower of the DTI-based eligibility and the LTV-based eligibility
(`property_value x max_LTV_percent`) is the applicant's final maximum eligible
loan amount, whenever a property value has been declared. If no property
valuation has been submitted yet, LTV cannot be verified and the DTI-based
figure should be quoted as provisional, pending valuation.

## 7. Decision Outcomes

- **Approved**: requested amount is at or below the maximum eligible amount,
  credit score is Prime or Standard.
- **Conditionally approved**: requested amount exceeds the maximum eligible
  amount, or credit score is Sub-prime - a reduced amount, longer tenure, or a
  co-applicant is suggested to bridge the gap.
- **Declined**: credit score is below policy, or maximum affordable EMI is
  zero/negative.
