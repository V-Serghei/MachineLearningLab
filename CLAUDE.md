# Claude Working Rules

> Project-specific configuration for **MachineLearningLab**.

---

## Language

- **Respond to the user in Russian.**
- Write all project files, code, comments, README.md, and this file in **English**.

---

## Git Protocol

1. **No commits** — only the user makes git commits.
   Never run `git commit`, `git push`, `git reset`, or any destructive git command
   without an explicit user request.
2. **All git commands are executed by the user only** — provide commands as
   plain text, never run them via Bash/PowerShell.
3. **Suggest a commit message** — after every set of changes, provide a
   ready-made conventional-commit message for the user to copy.

---

## Documentation

### README.md
- Contains **project description only**: what it is, tech stack, quick start,
  architecture overview.
- Must be updated after every significant code change.
- Rules and instructions for Claude go **only** in CLAUDE.md, never in README.md.

### CLAUDE.md (this file)
- Contains Claude's working rules, project context, and conventions.
- Add new rules and agreements here as they arise.
- Keep it up to date; stale rules are worse than no rules.

### .gitignore
- Must cover: IDE files (`.idea/`, `.vscode/`), Python cache (`__pycache__/`,
  `*.pyc`, `.ipynb_checkpoints/`), virtual environments (`.venv/`), OS files
  (`.DS_Store`, `Thumbs.db`), extracted large datasets (`data/creditcard.csv`).
- Maintain it continuously — add entries whenever new tool output or large files
  appear in the working tree.

---

## start.bat / stop.bat

**Both files are at the repository root.**

### start.bat responsibilities
1. Check Python is on PATH.
2. Create `.venv` virtual environment if absent.
3. Activate `.venv` and install dependencies.
4. Extract `data/creditcard.tar.xz` to `data/creditcard.csv` if the CSV is missing.
5. Run a quick import smoke test to verify the environment.

### stop.bat responsibilities
No persistent services — prints a message and exits.

### Maintenance rule
**Keep start.bat always accurate.** After adding a new dependency, update the
install command in the same changeset.

---

## Project Context

**MachineLearningLab** is an educational Python project structured in three labs:

| Lab | Topic | Algorithms |
|-----|-------|------------|
| LB1 | Linear Regression | Linear regression (movie revenue) |
| LB2 | Supervised Classification | AdaBoost, CN2, CART, Complement NB, RBF+LogReg |
| LB3 | Unsupervised Clustering | K-Means, DBSCAN, GMM, Agglomerative |
| guide/ | Reference examples | Kernel SVM |

Each lab is a self-contained Python script with a matching `.ipynb` notebook.
Scripts use relative paths `../data/` to reach datasets.

---

## Code Style

- No speculative features, no abstractions beyond the current task.
- No comments explaining WHAT code does; only add one when the WHY is non-obvious.
- All source code, comments, and string literals must be in **English**.
- No trailing summaries in responses — the user can read the diff.
