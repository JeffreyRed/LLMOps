# 🔐 Setting Up CI Secrets on GitHub

Your API keys must be added to GitHub as **Secrets** — they are encrypted and
never visible in logs. This is how CI accesses them without you ever committing them.

---

## Steps

### 1. Go to your repo on GitHub
`https://github.com/YOUR_USERNAME/agentic-rag-pipeline`

### 2. Navigate to Settings → Secrets
```
Your repo → Settings → Secrets and variables → Actions → New repository secret
```

### 3. Add these two secrets exactly:

| Name | Value |
|---|---|
| `GROQ_API_KEY` | your key from console.groq.com |
| `LANGCHAIN_API_KEY` | your key from smith.langchain.com |

---

## What happens after every `git push`?

```
git push origin main
        │
        ▼
GitHub Actions triggers automatically
        │
        ├── Job 1: lint       → checks code style (Black + Flake8)
        │         ↓ pass
        ├── Job 2: test       → runs pytest unit tests
        │         ↓ pass  
        └── Job 3: eval       → runs evaluate.py, checks score ≥ 0.35
                  ↓ pass
              ✅ All green — safe to merge!
              ❌ Any fail — GitHub blocks the merge and emails you
```

---

## Viewing results

Go to your repo → **Actions tab** → click any workflow run to see:
- Which job failed and why
- Full logs line by line
- Downloaded eval report JSON artifacts

---

## Badge for your README

Add this to the top of your README.md to show CI status:

```markdown
![CI](https://github.com/YOUR_USERNAME/agentic-rag-pipeline/actions/workflows/ci.yml/badge.svg)
```

It turns green when passing, red when failing. Looks very professional.