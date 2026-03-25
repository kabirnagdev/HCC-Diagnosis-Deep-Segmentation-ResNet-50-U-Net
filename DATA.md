# Data Governance and Usage

## Dataset Source

Primary reference dataset cited by project materials:

- Liver Tumor Segmentation Dataset (Kaggle)

## Data Handling Principles

- Do not commit protected health information (PHI).
- Keep raw medical scans out of public git history when licensing/privacy is unclear.
- Use `.gitignore` and external storage for large/private artifacts.

## Recommended Data Layout

Create local folders (not versioned) such as:

- `data/raw/`
- `data/processed/`
- `data/splits/`

## Ethical and Legal Compliance

Users are responsible for compliance with institutional review policies, dataset terms, and applicable regulations.
