---
title: Engineering Runbook
access: engineering
---
# Engineering Runbook

To roll back a faulty release run the rollback command with the previous version flag, then confirm health checks are green.

Database backups run nightly at 02:00 UTC and are retained for 30 days. Restore drills happen every quarter.

The on-call rotation changes weekly on Monday morning. Severity one incidents must page the on-call engineer immediately.
