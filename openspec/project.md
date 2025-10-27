# Project Context

## Purpose
OpenSpec is a lightweight, AI-native system for spec-driven development. It aligns humans and AI coding assistants on what to build before any code is written, ensuring deterministic and reviewable outputs without requiring API keys.

## Tech Stack
- **Language:** TypeScript
- **Runtime:** Node.js (>= 20.19.0)
- **Testing:** Vitest
- **CLI Framework:** Commander.js
- **Schema Validation:** Zod
- **Release Management:** Changesets

## Project Conventions

### Code Style
The project follows standard TypeScript best practices. Code should be clean, readable, and well-documented where necessary.

### Architecture Patterns
OpenSpec uses a two-folder model to separate the source of truth from proposed changes:
- `openspec/specs/`: Contains the current, authoritative specifications.
- `openspec/changes/`: Holds proposed updates, tasks, and spec deltas for new features or modifications.

The application is a command-line interface (CLI) built with Commander.js.

### Testing Strategy
The project uses Vitest for unit and end-to-end testing. Tests are located in the `test/` directory and are run with the `pnpm test` command. The testing strategy emphasizes ensuring the reliability and correctness of the CLI commands and core logic.

### Git Workflow
The project follows the Conventional Commits specification for commit messages. The `changeset` tool is used to manage versioning and generate changelogs for releases.

## Domain Context
The core concepts of OpenSpec are:
- **Specs:** The source of truth for what the system should do.
- **Changes:** Proposals for new features or modifications, containing spec deltas and implementation tasks.
- **Proposals:** The initial description of a change.
- **Tasks:** A checklist of implementation steps for a change.
- **Archiving:** The process of merging a completed change into the authoritative specs.

## Important Constraints
- The project requires Node.js version 20.19.0 or higher.

## External Dependencies
OpenSpec integrates with a variety of AI coding assistants, including:
- Claude Code
- CodeBuddy
- Cursor
- and others that support the AGENTS.md convention.