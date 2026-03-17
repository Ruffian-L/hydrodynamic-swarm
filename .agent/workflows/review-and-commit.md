---
description: How to run Jules research documentation and CodeRabbit review before committing
---

# Post-Change Workflow: Jules + CodeRabbit

After making changes, run this workflow before committing. Both tools run locally — no invisible GitHub Actions.

## 1. Jules Research Log (after significant changes)

Run Jules locally to analyze your changes and write a research log entry:

```bash
jules review --diff HEAD
```

Jules will create/update entries in `research_logs/` following the template. Review what Jules wrote before committing.

## 2. CodeRabbit Review (before every commit)

The pre-commit hook runs this automatically, but you can also run it manually:

```bash
coderabbit review --staged
```

This reviews your staged changes and blocks if there are issues.

## 3. Commit (requires YubiKey touch)

```bash
git add -A
git commit -S -m "your commit message"
```

The `-S` flag signs with your GPG key on the YubiKey. You'll need to physically touch the key.

The pre-commit hook will run: clippy → bless → audit → coderabbit.

## 4. Push (requires YubiKey touch for SSH)

```bash
git push origin master
```

Branch protection on GitHub requires signed commits. Unsigned commits will be rejected.
