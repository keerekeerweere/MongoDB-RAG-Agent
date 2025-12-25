# AGENTS.md – Repository Configuration Guide for MongoDB RAG Agent

## Purpose & Scope
This document defines build and test workflows; coding standards; Cursor/Copilot rule integration requirements that agentic tools must follow.

## 1. Build / Lint / Test Commands  
### Dependency Installation  
```
uv sync
```

### Formatting (Black)  
```
black --line-length 88 .
```

### Linting (Ruff)  
```
ruff check --fix
```

### Type Checking (Mypy)  
```
mypy --ignore-missing-imports --strict-optional . --non-interactive
```

### Test Execution Overview  
General form: `pytest -xvs --tb=short {test_path}`  

#### Single‑Test Shortcut Pattern  
```bash
pytest -k "test_name" --maxfail=1 --disable-warnings --tb=short test_scripts/{file}.py
```

*Example:* Run search test only:
```
pytest -k "search" --maxfail=1 --disable-warnings --tb=short test_scripts/test_search.py
```

### CI Snippet Example (GitHub Actions)  
```yaml
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - name: Run single critical test
        run: |
          pytest -k "test_search" --maxfail=1 --disable-warnings --tb=short test_scripts/test_search.py
      - name: Lint modified staged files  
        run: |
          git diff --cached --name-only | xargs ruff check --fix
```

## 2. Code Style Guidelines  

### Import Management (Ordered & Grouped)  
```python
# Standard Library (alphabetical)
import json
import logging
import pathlib  

# Third-Party Dependencies (grouped by origin)  
import numpy as np  
import pandas as pd  

# Local Modules (relative imports, preserve hierarchy)  
from .core.agent_manager import AgentManager  
from ..ingestion.chunker import Chunker
```

### Naming Conventions  
| Category | Convention |
|---------|------------|
| Constants | `UPPER_SNAKE_CASE` (e.g., MAX_RETRIES=3) |
| Functions / Variables | `snake_case` |
| Classes / Exceptions | `PascalCase` |
| Custom Exceptions | Singular nouns (`InvalidQueryParameterError`) |

### Type Hinting Rules  
- **Public APIs required** – all module, class, function signatures public-facing must include hints  
- **Private helpers optional** – omit type hints only when prefixed with underscore `_helper()`  
- Use `Literal`/typed dictionaries for enumeration patterns:  
  ```python
  from typing_extensions import Literal  
  Status = Literal["INIT", "RUNNING", "COMPLETED"]
  ```

### Line Length & Wrapping  
- Maximum line width: **88 characters**  
- Continue multi‑line statements using parentheses; avoid backslash continuation

### Error Handling Protocol  
```python
raise NotImplementedError(f"Missing implementation: {__file__}:{lineno}")
```  
- Include file name and exact line number context with exception message  
- Never raise unannotated `Exception` or bare error classes  

### Logging Best Practices  
```python
BASE_LOGGER = logging.getLogger(__name__)  
logging.basicConfig(level=logging.INFO)  
BASE_LOGGER.error("Failed operation: %s", exc_info=True)  # Always capture stack trace
```

### Property Definition Rules  
Public `@property` must be explicitly typed and include descriptive docstring:
```python
@property  
def metadata(self) -> dict[str, str]:  
    """Return static agent metadata."""  
    return {"version": "1.2.0", "status": "ACTIVE"}
```

### Custom Exception Naming  
Create one‑word singular noun names ending in `Error`:
- `ConfigurationNotFoundError`
- `ResourceUnavailableError`

## 3. Cursor / Copilot Rule Integration (if files exist)  
**Cursor Rules:** Check `.cursor/rules/*.md` or fallback to `.cursorrules`. Include content verbatim:  
```
<!-- BEGIN CURSOR_RULES --><!-- END CURSOR_RULES -->
```  

**Copilot Instructions:** If `.github/copilot-instructions.md` exists, include section:  
```
<!-- BEGIN COPILOT_INSTRUCTIONS --><!-- END COPILOT_INSTRUCTIONS -->
```

## 4. Agent Development Checklist (For PRs / Commits)
- [ ] Verify single‑test execution succeeds via shortcut commands  
- [ ] Ensure full test suite passes with `coverage xml` reporting no gaps  
- [ ] Confirm linting and type checking complete without errors  
- [ ] Validate import ordering rules against `.isort.cfg` configuration  
- [ ] Apply naming conventions consistently across new/modified modules  
- [ ] Update Cursor/Copilot rule blocks for any newly added policies  

## 5. Commit & PR Standards  
### Commit Message Template (`type(scope): description`)  
```
feat(ingest): enable PDF text extraction for scanned documents  
fix(mongo_client): handle connection timeout edge case gracefully
```

### Pull Request Summary Structure  
- **Title:** concise benefit statement  
- **Description:** bullet points covering motivation, impact assessment, migration notes  
- **Related Issues:** link issue numbers or descriptions  
- **Checklist:**
  - [ ] Tests pass locally (`pytest -q`)  
  - [ ] Lint & mypy succeed without warnings or failures  
  - [ ] Documentation updated if relevant to change  
  - [ ] Cursor/Copilot rule blocks incorporated if scope affected them  

## Version History (excerpt)  
| Version | Date       | Summary |
|---------|------------|---------|
| v1.0    | 2025‑01‑01 | Initial release of AGENTS.md      |
| v1.1    | 2025‑03‑15 | Added single‑test execution example |
| v1.2    | 2025‑06‑27 | Integrated Cursor/Copilot rule templates |
| v1.3    | 2025‑12‑25 | Updated to use Ruff instead of Flake8 |

## Glossary  
- **CI:** Continuous Integration  
- **LGTM:** Look‑ahead Test Management (internal)  
- **RAG:** Retrieval‑Augmented Generation  

---  
*Document maintained at repository root; update via pull request when scope evolves.*