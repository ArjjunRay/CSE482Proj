# Security Policy

Security fixes are applied to the latest release on the default branch.

Do not open a public issue for a vulnerability or exposed credential. Use GitHub's
private vulnerability reporting feature on the repository's **Security** tab. Include
the affected version, reproduction steps, impact, and any suggested fix, but never
include live credentials or personal data.

The application reads `SPORTSDATAIO_KEY` from the environment. If a key is committed,
revoke it immediately and rewrite the affected Git history; deleting it in a later
commit is not sufficient.

