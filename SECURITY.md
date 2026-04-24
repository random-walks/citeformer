# Security policy

## Supported versions

citeformer is pre-1.0. Security fixes are applied to the latest released minor (currently `0.1.x`) and backported to older releases only on a best-effort basis.

| Version | Supported |
|---------|-----------|
| 0.1.x   | ✅         |
| < 0.1   | ❌         |

## Reporting a vulnerability

**Please don't open a public GitHub issue.** Security reports go to the maintainer directly so that a patch can land before the problem is disclosed.

- **Email**: <blaise@ubik.studio>
- **Subject line**: `[security] citeformer — <short description>`

Include in your report:

1. A description of the vulnerability and its impact.
2. Steps to reproduce (ideally a minimal Python script).
3. The citeformer version and Python version you're on (`uv run citeformer version`).
4. Any mitigations you've already identified.

## What happens next

- **Within 72 hours**: acknowledgement of receipt.
- **Within 7 days**: initial triage + severity assessment. If the report is valid, we'll agree on a disclosure timeline (typically 30–90 days depending on severity).
- **Coordinated disclosure**: a patch ships as a patch-version bump on PyPI; credit is given in the release notes and CHANGELOG if the reporter wants it.

## Threat model notes

What citeformer *claims* to protect against:

- **Citation fabrication** — marker tokens pointing at a non-existent source are structurally impossible to generate (local logit-layer backends) or to return in a validated payload (schema-layer API backends). See [guarantees.md](docs/guarantees.md).

What it does *not* protect against and should not be relied on for:

- **Prompt injection** inside retrieved `Source.content`. If an attacker controls a retrieved passage, they can influence the model's output text — they cannot, however, get the model to emit an out-of-scope cite id. Treat retrieved content with the same skepticism you'd treat any user-controlled input.
- **Supply-chain attacks on dependencies.** We pin minimum versions (not maximums) and rely on PyPI / upstream security. If a vulnerability affects a dep (xgrammar, transformers, etc.), ping us so we can bump the floor in a patch release.
- **Claim correctness.** `verify()` runs NLI entailment against the retrieved source — it doesn't verify that the source itself is correct. Garbage-in, cited-garbage-out.

## Hall of fame

Security reporters who want public credit appear here after their report is patched and disclosed.

*(Empty so far — citeformer is pre-1.0 and the attack surface is small. You could be the first.)*
