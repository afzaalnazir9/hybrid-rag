# Nimbus Notes — Troubleshooting and Error Codes

> Nimbus Notes is a fictional note-taking app invented for this demo.

## Sync errors

**NB-1001 — Authentication expired.** Your sign-in session has ended, usually after 60 days without opening the app. Sign out and sign in again. Unsynced changes are kept in the local queue and will upload after you sign in.

**NB-1004 — Workspace limit reached.** You tried to create an eleventh workspace on the Free plan. Delete a workspace or upgrade to Pro.

**NB-2210 — Attachment too large.** The file is larger than your plan allows (25 MB on Free, 250 MB on Pro and Team). Compress the file or store it elsewhere and paste a link.

**NB-3302 — Sync conflict could not be merged.** Two devices changed the same note in ways Nimbus could not combine automatically. Both versions are kept; look for the note with "(conflicted copy)" in its title and merge the text by hand.

**NB-4012 — Local database corrupted.** The app's local cache on this device is damaged, often after the device ran out of disk space during a sync. Your notes on the server are safe. Go to Settings → Advanced → "Rebuild local cache". The rebuild downloads every offline notebook again and can take several minutes on large accounts.

**NB-5000 — Server unavailable.** The Nimbus servers could not be reached. Check the status page at status.nimbus.example. The app retries automatically every 2 minutes.

## The app is slow

Large notebooks (more than 5,000 notes) can make the sidebar slow to open. Split them into smaller notebooks, or turn off "Show note previews" in Settings → Appearance. Turning off "Index PDF text" in Settings → Search also reduces CPU use on older computers, but PDF contents will no longer appear in search results.

## Notes are missing

1. Check the Trash. Deleted notes stay there for 30 days before they are removed permanently.
2. Check that you are in the right workspace; the workspace switcher is at the top of the sidebar.
3. If a notebook was shared with you, the Owner may have removed your access.
4. If none of the above helps, restore the note from Version history, or contact support.

## Contacting support

Free users can get help in the community forum. Pro users can email support and receive a reply within 2 business days. Team plan customers get priority support with a reply within 4 business hours and can also book a call with an account manager.

When you contact support, include the error code (for example NB-4012), your operating system, and the app version shown in Settings → About.
