# Nimbus Notes — Plans, Billing and Privacy

> Nimbus Notes is a fictional note-taking app invented for this demo.

## Plans

| | Free | Pro | Team |
|---|---|---|---|
| Price | $0 | $6 per month, or $60 per year | $10 per user per month |
| Workspaces | Up to 10 | Unlimited | Unlimited |
| Attachment size | 25 MB | 250 MB | 250 MB |
| Storage | 2 GB | 100 GB | 1 TB per user |
| Version history | 7 days | 90 days | 1 year |
| Devices | 2 | Unlimited | Unlimited |
| Shared notebooks | No | No | Yes |
| Support | Community forum | Email, 2 business days | Priority, 4 business hours |

Students and teachers with a school email address get Pro for $3 per month. Non-profit organisations get a 50% discount on the Team plan.

## Billing

Subscriptions renew automatically at the end of each billing period. You can cancel at any time in Settings → Billing; your plan stays active until the end of the period you have paid for. Annual Pro subscriptions can be refunded in full within 14 days of purchase. Monthly subscriptions are not refunded.

If a payment fails, Nimbus retries three times over 10 days. If all retries fail, the account moves to the Free plan. Nothing is deleted, but notes over the Free storage limit become read-only until storage is reduced or the subscription is renewed.

## Device limit on the Free plan

The Free plan can be used on 2 devices at the same time. Signing in on a third device asks you to sign out of one of the others. Web browser sessions do not count toward the device limit.

## Privacy and security

Notes are encrypted in transit with TLS 1.3 and at rest with AES-256. Team plan administrators can turn on single sign-on (SSO) with SAML 2.0 and can require two-factor authentication for every member.

Pro and Team users can turn on end-to-end encryption for individual notebooks. End-to-end encrypted notebooks cannot be searched from the web app, and if you lose your encryption passphrase, Nimbus cannot recover those notes.

Nimbus does not sell personal data and does not use the contents of your notes to train machine-learning models. Account data is stored in data centres in the EU or the US, chosen when the workspace is created; it cannot be moved later.

## Deleting your account

Go to Settings → Account → "Delete account". All notes, attachments and version history are permanently erased within 30 days. Export your notes first with Settings → Export, which produces a ZIP file of Markdown files and attachments.
