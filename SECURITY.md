# Security Policy

## Supported versions

| Version | Supported |
|---------|-----------|
| 1.x     | ✅ Yes    |

## Reporting a vulnerability

**Please do not open a public GitHub issue for security vulnerabilities.**

Report security issues by emailing **mfarooqshafee333@gmail.com**.  
You should receive a response within **48 hours**. If not, follow up.

Please include:
- A description of the vulnerability and its potential impact
- Steps to reproduce (proof-of-concept if possible)
- Affected version(s)
- Any suggested mitigations

We follow [responsible disclosure](https://cheatsheetseries.owasp.org/cheatsheets/Vulnerability_Disclosure_Cheat_Sheet.html): we will acknowledge receipt, work on a fix, coordinate a release, and credit you if you wish.

## Security design notes

For anyone auditing this project:

| Control | Implementation |
|---------|----------------|
| Authentication | Bearer token (`API_KEY` env var) on all write endpoints (`/ingest`) |
| Path traversal | `/ingest` validates `source_path` is inside `ALLOWED_INGEST_DIR` |
| Security headers | `X-Content-Type-Options`, `X-Frame-Options`, `X-XSS-Protection`, `Referrer-Policy`, `Cache-Control: no-store` on every response |
| CORS | Configurable via `CORS_ORIGINS`; defaults to `*` only for local dev |
| Input validation | All request bodies parsed and bounded by Pydantic models |
| Secret management | All secrets via environment variables; no secrets in source code |
| Dependency scanning | GitHub Dependabot alerts enabled |
