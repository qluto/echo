# Changelog

The app's "What's new" screen and the GitHub release notes show this file as is.
Before each release, rename `## Unreleased` to `## vX.Y.Z — YYYY-MM-DD`, then tag (the release fails without the section).

## Unreleased

## v0.16.0 — 2026-11-16

### Added
- Choose how long a conversation must go on before automatic meeting detection starts a meeting: 1–4 minutes (Settings → Meeting Detection). The other detection conditions scale with it.
- When automatic meeting detection starts a meeting, a notice appears just below Echo's menu bar icon, so you notice it even while working on another desktop.
- Check for updates from the menu bar menu and from Settings.
- After an update, the menu bar notice links to what's new. Settings → About shows the full changelog.

## v0.15.0 — 2026-11-15

### Added
- Sounds when you press the hotkey and when the transcript is ready.

### Changed
- Removed always-on transcription (the Log tab). Automatic meeting detection records conversations instead. Your existing history stays in the database but is no longer shown in the app.
- Removed the voice memo recorder in favor of Meetings, and renamed the tab to "Meetings". Earlier voice memos open from the meeting list.

## v0.14.0 — 2026-11-14

### Added
- Automatic meeting detection (optional): once a conversation goes on for 3 minutes, Echo starts recording a meeting from the beginning of the conversation.

## v0.13.4 — 2026-11-12

### Added
- Local transcription API (optional): other apps on this Mac can use Echo's transcription over HTTP and WebSocket streaming.

## v0.13.3 — 2026-11-11

### Added
- Automatic updates from GitHub Releases.

For earlier changes, see the [GitHub releases](https://github.com/qluto/echo/releases).
