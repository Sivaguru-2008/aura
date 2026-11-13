# Contributing to AURA

Thank you for your interest in contributing to **AURA** (Adaptive Uncertainty-aware Reasoning Assistant).

AURA is an open-science medical AI platform and clinical reproducibility audit harness. We welcome contributions that improve clinical safety, mathematical rigor, evaluation tooling, and software performance.

---

## 1. Code of Conduct & Open Science Principles

Our research is grounded in transparent, pre-registered evaluation. We adhere strictly to the following principles:
- **No Selective Reporting:** Benchmarks must be reported across all evaluated configurations and pre-registered cohorts without post-hoc cherry-picking.
- **Compute and Optimization Normalization:** Any comparator evaluation must match training budgets, hyperparameter sweeps, and learning schedules fairly.
- **Cryptographic Provenance:** Datasets and derived intermediate tensors must be pinned via SHA-256 in verifiable ledgers.
- **Zero Data Leakage:** Splits must strictly be patient-disjoint with zero cross-split overlap.

---

## 2. Development Setup

Clone the repository and set up a virtual environment:

```bash
git clone https://github.com/Sivaguru-2008/aura.git
cd aura
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

---

## 3. Verification & Testing

Every Pull Request must pass automated verification:

### Level A: Reproduction & Drift Tests (<15 seconds)
Verify bit-for-bit mathematical consistency and ensure no published claim has drifted from its underlying serialized artifact:
```bash
python reproduce_paper.py
```

### Level B: Comprehensive Test Suite (~1-2 minutes)
Run the complete automated pytest suite:
```bash
pytest aura/tests/ -v
```

### Level C: DeLong / McNemar / Bootstrap Significance Audits
```bash
python audit_all.py
```

---

## 4. Submitting Pull Requests

1. **Create a Topic Branch:**
   ```bash
   git checkout -b feat/your-feature-name
   ```
2. **Follow Code Conventions:**
   - Follow standard PEP 8 conventions.
   - Maintain type annotations using `typing` / `pydantic`.
   - Ensure all public functions and clinical scoring routines have comprehensive docstrings explaining biological/clinical assumptions.
3. **Commit Cleanly:**
   - Use conventional commit messages: `feat:`, `fix:`, `docs:`, `test:`, `chore:`.
4. **Push and Open a Pull Request:**
   - Link any relevant issues or preregistration items.
   - Attach logs showing that `python reproduce_paper.py` and `pytest` passed.

---

## 5. Security & Responsible Clinical Disclosure

If you discover a security vulnerability or potential clinical data leakage risk, please do not open a public issue. Review our [SECURITY.md](docs/SECURITY.md) policy or email the project maintainer directly at **sivagurumurugan1@gmail.com**.
