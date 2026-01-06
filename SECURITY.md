# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

**Do not report security vulnerabilities through public GitHub issues.**

Instead, please report security vulnerabilities by emailing:

**security@goodai.com**

Please include:

1. Description of the vulnerability
2. Steps to reproduce
3. Potential impact
4. Suggested fix (if any)

### What to Expect

- **Acknowledgment**: Within 48 hours
- **Initial Assessment**: Within 7 days
- **Resolution Timeline**: Depends on severity
  - Critical: 7 days
  - High: 14 days
  - Medium: 30 days
  - Low: 90 days

### Disclosure Policy

- We follow coordinated disclosure
- Credit will be given to reporters (unless anonymity is requested)
- We will notify you when the vulnerability is fixed

## Security Best Practices

When using this framework:

1. **Never commit secrets** - Use environment variables for API keys, passwords
2. **Validate inputs** - Use the provided validation utilities
3. **Enable audit logging** - Track access and changes in production
4. **Use RBAC** - Implement role-based access control for multi-user deployments
5. **Keep dependencies updated** - Enable Dependabot alerts

## Known Limitations

- This is v0.1.0 - not battle-tested at enterprise scale
- The framework does not provide encryption at rest
- Authentication is not included - integrate with your identity provider
