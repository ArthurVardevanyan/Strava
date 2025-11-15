# Strava Activity Commands (WIP)

Some Commands put together for migration purposes

This repository contains `strava.py`, a CLI utility that lists your Strava activities, enriches them with location details, and optionally saves activity files (GPX/TCX/original zip) organized by country/state/city.

## Requirements

- Python 3.10+ (or compatible runtime)
- The [`requests`](https://pypi.org/project/requests/) library (`pip install -r requirements.txt`)
- Valid Strava API credentials (`client_id`, `client_secret`, `refresh_token`). See `strava.py --print-auth-url` for help obtaining them.

## Setup

1. Store your credentials in environment variables (`STRAVA_CLIENT_ID`, `STRAVA_CLIENT_SECRET`, `STRAVA_REFRESH_TOKEN`) or in a `strava_secrets.json` file next to `strava.py`.
2. Install dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

## Usage highlights

```bash
python .\strava.py --types Walk,Hike --geocode --download --download-format gpx --download-dir .\downloads
```

This will:

- List Walk/Hike activities, optionally enriched with detail calls and reverse geocoding.
- Download each activity in GPX format.
- Save each file under `downloads/<activity-type>/<country>/<state>/<city>/<activity-id>.gpx`, using `unknown_*` fallbacks when location information is missing.

Use `--download-quiet` to mute per-file download logs once you confirm the downloads are working.

If Strava rejects the browser export URL (HTTP 403), the script now falls back to fetching activity streams and generating a GPX file locally, so `--download-format gpx` still succeeds even without the web export permissions.

The download step also caches files on disk. If a file already exists at the expected `country/state/city/<activity-id>.<ext>` path, the script skips the API call and keeps the existing file untouched (unless you delete it first).

The script pre-scans the download directory before contacting Strava, so rerunning it doesn't just skip the downloads but also avoids any extra detail/geocode requests for those already-exported activities.

Once every matching activity on a page has already been exported, the downloader stops requesting further pages—so you won't keep hitting the API for older activities that you already imported.

See `python .\strava.py --help` for the complete list of CLI options (CSV export, pagination, progress bar, detail API caps, etc.).

## Run

```bash
python .\strava.py --location-details --detail-sleep-ms 200 --types Walk,Hike  --geocode --geocode-sleep-ms 250 --progress-bar  --download --download-format gpx --test

python .\strava.py --rescan-unknown --geocode --geocode-sleep-ms 250 --progress-bar --test

python strava.py --retro-convert-fit
```

## REF

Primary documentation and resources referenced while building this script:

- Garmin FIT Protocol: <https://developer.garmin.com/fit/protocol/>
- Garmin FIT SDK (for field numbers, CRC reference): <https://developer.garmin.com/fit/sdk/>
- Strava V3 API Overview: <https://developers.strava.com/>
- Strava Rate Limits: <https://developers.strava.com/docs/rate-limits/>
- Strava Activity Streams: <https://developers.strava.com/docs/reference/#api-Streams>
- GPX 1.1 Schema: <https://www.topografix.com/gpx.asp>
- TCX (Training Center XML) Schema: <https://www8.garmin.com/xmlschemas/TrainingCenterDatabasev2.xsd>
- FIT File Basics / Reverse Engineering Notes (community): <https://www.thisisant.com/forum/viewthread/4492/> (ANT+ forums)
- OpenStreetMap Nominatim API (reverse geocoding): <https://nominatim.org/release-docs/latest/api/Reverse/>
- Haversine formula reference (distance between lat/lon points): <https://en.wikipedia.org/wiki/Haversine_formula>
- Python datetime ISO parsing: <https://docs.python.org/3/library/datetime.html>
- Python requests library: <https://requests.readthedocs.io/>
- Semicircle coordinate conversion (FIT lat/lon): <https://forums.garmin.com/developer/fit-sdk/f/discussion/> (various posts)

If you extend support for other sports or add developer data messages, also consult:

- FIT Profile CSVs (sport/subsport enums): included in FIT SDK downloads
- Developer data fields guide: <https://developer.garmin.com/fit/protocol/#developerData>
