---
created: 2026-02-03T16:50
title: Cat heredoc pattern doesn't work in Claude Code
area: tooling
files: []
---

## Problem

The `cat > file << 'EOF'` pattern commonly used to create files inline doesn't work in Claude Code's bash environment. This affects verification workflows where test config files need to be created temporarily.

Workaround: Use Write tool to create files, provide bash commands separately, then delete files after.

## Solution

This is a Claude Code environment constraint, not a project issue. Document the workaround pattern for future sessions:
1. Write files using Write tool
2. Provide commands to run
3. Delete files after verification
