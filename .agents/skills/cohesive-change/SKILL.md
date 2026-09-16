---
name: cohesive-change
description: Maintain one coherent implementation path when modifying code, tests, or executable and verification scripts in an existing repository. Use for features, bug fixes, refactors, and follow-up edits that could introduce duplicate owners, fallback paths, compatibility shims, or layered patches. Do not use for documentation-only edits or generated artifacts.
---

# Cohesive Change

Keep one canonical implementation path for each behavior. Judge a change by the number of competing owners and execution paths it leaves behind, not by its file count.

## Establish the Change Contract

Before editing:

1. Read the current repository instructions, architecture documentation, and relevant code. Treat them as authoritative over conversation memory.
2. Record `git status --short` as the worktree baseline. If a target file is already dirty, preserve its pre-task diff in tool output or a temporary location outside the repository so the task delta remains distinguishable.
3. Trace the existing behavior through its callers, configuration, tests, and outputs.
4. Identify the single module that should own the behavior, the invariants that must remain true, and any path the change should replace.

Keep this contract in the working context. Do not create planning, manifest, handoff, or log files unless the user requested them as deliverables.

## Change the Owner

Correct or extend the module that owns the behavior. Do not avoid that change by adding:

- a parallel `new`, `v2`, or `legacy` implementation;
- a fallback that silently keeps both implementations active;
- duplicate configuration or state for the same decision;
- caller-specific exceptions that encode the same policy in several places;
- a wrapper or compatibility shim without an explicit compatibility requirement.

When a new path replaces an old one, migrate its callers and remove the superseded code, configuration, and tests in the same task. Legitimate boundary adapters may remain when they express distinct interfaces rather than duplicate policy.

Perform consolidation that is within the user's requested scope without pausing. If the coherent solution requires a material scope expansion, explain the ownership conflict and ask the user to choose; do not install a workaround while waiting.

Do not turn this rule into opportunistic cleanup. Leave unrelated architecture and behavior untouched. A necessary multi-file change is acceptable when every file participates in the same ownership boundary.

## Review the Task Delta

Before declaring completion, compare the final worktree with the recorded baseline and inspect every task-owned change, including untracked files. In a dirty worktree, review only the task delta and preserve all pre-existing user changes.

Resolve these findings before completion:

- two modules now own the same policy, state, or transformation;
- a new conditional is bolted around the canonical flow instead of changing it;
- obsolete functions, flags, configuration, tests, or files remain after replacement;
- the change introduces another representation that must be kept synchronized;
- a new abstraction or file has no durable ownership role.

Run verification appropriate to the affected contract. Tests should exercise public behavior and invariants rather than merely mirror new branches.

The change is incomplete while superseded implementation paths remain, unless the user explicitly requires compatibility or the necessary consolidation is outside the authorized scope. Report either exception clearly instead of describing deferred cleanup as complete.
