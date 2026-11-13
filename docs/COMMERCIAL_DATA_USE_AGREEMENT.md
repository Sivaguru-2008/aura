# Commercial data use agreement: draft terms for counsel

> **A draft to discuss with a lawyer. It is not legal advice and not a signable agreement.**
> "[Company]" is a placeholder until the company is incorporated. Every point below is the
> company's opening position, and hospitals negotiate most of them.

**Parties:** a hospital or imaging provider ("Data Contributor") and [Company] ("Recipient").

## 1. Purpose

The Recipient gets data it may use to build, validate and sell AI software. That replaces
the research-only datasets the current models were trained on (see
[`DATA_LICENSES.md`](DATA_LICENSES.md)). The agreement has to name commercial product
development as a permitted purpose in so many words. A research agreement that is silent on
it does not cover it.

## 2. De-identification

1. **Standard.** Data is de-identified under the HIPAA Privacy Rule, 45 CFR § 164.514(b), by
   one of two methods:
   - **Safe Harbor.** Remove the 18 listed identifiers. These include every element of a date
     except the year, and ages over 89 must be grouped as "90 or older". Shifted dates are
     still date elements, so a dataset that keeps them is not Safe Harbor.
   - **Expert Determination.** A qualified expert certifies that the risk of re-identification
     is very small. This is the usual route when the model needs dates, intervals between
     studies, or finer geography.

   If identifiers such as dates must stay, the data may instead be a limited data set under
   45 CFR § 164.514(e). A limited data set is still PHI and needs its own data use agreement.
   Ask counsel whether commercial product development fits the purposes a limited data set
   allows.
2. **DICOM headers.** Follow the DICOM attribute confidentiality profiles (PS3.15 Annex E),
   including private tags.
3. **Burned-in text.** Detect and remove text burned into the pixels before images leave the
   provider.
4. **No re-identification.** The Recipient will not try to re-identify or contact any
   patient.

## 3. Rights in the data and in what is built from it

The opening position has three parts:
- The Recipient gets a non-exclusive, worldwide, perpetual licence to use the de-identified
  data to train, validate and improve its software.
- The Recipient owns the resulting weights and software.
- The Recipient may sell and deploy that software, including as a regulated medical device.

Hospitals commonly ask for the following. Settle each with counsel:
- a fee, royalty, revenue share or equity, or the product at a discount
- joint ownership of models or improvements, or a licence back to the hospital
- the right to publish results, with a review window for the Recipient
- no resale or onward transfer of the data, and no use beyond the agreed purpose
- return or destruction of the data when the term ends
- attribution

"Trained weights are not PHI" is an argument to put to counsel, not a clause to assume. It
depends on the model, and on whether memorised data can be extracted from it.

## 4. What data is needed: the Recipient's plan, not FDA requirements

FDA does not publish fixed minimum cohort sizes. A submission justifies its sample size
statistically, and an FDA Pre-Submission (Q-Sub) meeting is where that gets agreed. The
figures below are proposed targets for the Recipient's own development and validation plan.

| Track | Proposed size | Content | Diversity | Labels |
|---|---|---|---|---|
| Chest X-ray | about 10,000 patients | Frontal PA/AP DICOM | Several scanner vendors and sites | Radiologist consensus ([`RADIOLOGIST_ANNOTATION_PROTOCOL.md`](RADIOLOGIST_ANNOTATION_PROTOCOL.md)) |
| Brain MRI | to be agreed | FLAIR, T1, T1ce and T2 | 1.5 T and 3 T | Segmentation adjudicated by a neuroradiologist |
| Clinical metadata | where available | Age, sex, relevant labs | — | From the EHR |

Hold out a test set that no training or tuning step ever reads, from a different site if
possible.

## 5. Security and transfer

- **Transfer.** Encrypt in transit (TLS 1.2 or later), or ship encrypted media with chain of
  custody. The alternative is to keep the data on site and bring the training to it.
- **Controls.** Describe the controls that actually exist: encryption at rest, role-based
  access, audit logging and breach notification. [`SECURITY.md`](SECURITY.md) covers the
  software's own controls. SOC 2 or ISO 27001 certification is a later goal, not something a
  pre-seed company can promise today.
- **Business associate agreement (BAA).** A BAA is needed only if the Recipient receives PHI.
  Fully de-identified data does not need one. A limited data set needs a data use agreement.

## 6. Clauses counsel will add

- Term and termination
- Return or destruction of data
- Ethics or IRB approval where required
- Liability, indemnity and insurance
- Audit rights
- Governing law
- Signature blocks
