# RBI Regulatory Guidelines Referenced for Home Loan Underwriting

Summary of the Reserve Bank of India prudential norms this policy is built to
comply with. This is a reference summary for underwriting staff, not the full
regulatory text.

## Loan-to-Value (LTV) Ratio Slabs

Per RBI's risk-weight guidelines for individual housing loans, the maximum LTV
ratio a bank may extend is capped by the loan amount slab:

| Loan Amount Slab | Maximum LTV Ratio |
|---|---|
| Up to ₹30 lakh | 90% |
| Above ₹30 lakh and up to ₹75 lakh | 80% |
| Above ₹75 lakh | 75% |

LTV = (loan amount / property value) x 100. A lower LTV means the borrower is
contributing a larger down payment relative to the property's value.

## Risk Weights

Housing loans carry differentiated risk weights based on both the LTV ratio
and the loan amount slab, which affects the capital the bank must hold against
the loan but does not directly change individual applicant eligibility - it is
noted here only for underwriter context.

## Fair Practices in Lending

- Applicants must be informed of the exact interest rate, all applicable fees,
  and the basis on which their loan amount was determined.
- Rejections must be communicated with the specific reason (credit score,
  income, DTI/FOIR, or LTV).
- Interest rates offered must be linked to an external benchmark (repo-linked
  lending rate) plus a spread determined by the applicant's credit risk
  profile; see `interest_rate_card.csv` for the bank's current spread by
  credit score band.
