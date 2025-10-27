# Proposal: Archive All Completed Changes

## 1. Summary

This proposal is to add a new feature to the `openspec` command-line tool that will allow you to archive all completed changes at once, instead of having to archive them one by one.

## 2. The Problem

Currently, if you have multiple completed changes, you have to run the `openspec archive` command for each one. This is repetitive and time-consuming.

## 3. The Solution

I will add a new `--all` option to the `openspec archive` command.

When you run `openspec archive --all`, the tool will automatically find all the changes that are finished and archive them for you.

## 4. How it Will Work: Specification

The `archive` command will be updated as follows:

*   **To archive a single change (existing functionality):**
    *   You will still be able to run `openspec archive <change-name>`.
*   **To archive all completed changes (new functionality):**
    *   You will be able to run `openspec archive --all`.

## 5. Implementation Plan

Here is the plan for how I will build this feature:

### Step 1: Update the Core Logic
*   Modify the `ArchiveCommand` to understand the new `--all` option.
*   Add logic to find all the active changes.
*   For each change, check if all its tasks are completed.
*   If a change is complete, archive it.

### Step 2: Update the Command-Line Interface (CLI)
*   Add the `--all` option to the `archive` command.

### Step 3: Write Tests
*   Create a new test to make sure the `openspec archive --all` command works correctly.
*   The test will check that all completed changes are archived, and that incomplete changes are not.