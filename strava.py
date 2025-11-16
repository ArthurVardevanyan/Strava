#!/usr/bin/env python3

from __future__ import annotations

# Standard library imports
import sys, os, csv, json, time, re, itertools, zipfile, argparse
import datetime as dt
from typing import Any, Dict, List, Optional, Set, Tuple
import argparse
import csv
import datetime as dt
import json
import os
import re
import sys
import time
import itertools
import xml.etree.ElementTree as ET
import zipfile
from typing import Dict, Generator, Iterable, List, Optional, Set, Tuple

import requests
import xml.etree.ElementTree as ET
try:
	from zoneinfo import ZoneInfo  # Python 3.9+
except Exception:
	ZoneInfo = None  # Will fallback if missing

SCRIPT_ROOT = os.path.dirname(os.path.abspath(__file__))


STRAVA_TOKEN_URL = "https://www.strava.com/oauth/token"
STRAVA_ACTIVITIES_URL = "https://www.strava.com/api/v3/athlete/activities"
STRAVA_AUTH_URL = "https://www.strava.com/oauth/authorize"
STRAVA_ACTIVITY_DETAIL_URL_TMPL = "https://www.strava.com/api/v3/activities/{id}"
STRAVA_ACTIVITY_STREAMS_URL_TMPL = "https://www.strava.com/api/v3/activities/{id}/streams"
STRAVA_DOWNLOAD_URLS = {
	"gpx": "https://www.strava.com/activities/{id}/export_gpx",
	"tcx": "https://www.strava.com/activities/{id}/export_tcx",
	"original": "https://www.strava.com/activities/{id}/export_original",
}

def wait_to_next_quarter() -> None:
	"""Sleep until the next 0/15/30/45 minute slot."""
	now = dt.datetime.now(dt.timezone.utc)
	remainder = now.minute % 15
	minutes_to_add = 15 - remainder if remainder != 0 else 15
	target = now + dt.timedelta(minutes=minutes_to_add)
	target = target.replace(second=0, microsecond=0)
	wait_seconds = (target - now).total_seconds()
	print(f"Rate limited, waiting until {target.isoformat()} ({int(wait_seconds)}s)", flush=True)
	time.sleep(wait_seconds)


def rescan_unknown_locations(args) -> int:
	"""Refresh cached downloads stuck under unknown_* folders using local exports and optional geocoding."""
	unknown_files = find_unknown_location_files(args.download_dir)
	if not unknown_files:
		print("No files found in unknown location folders.")
		return 0
	geocode_cache: Dict[str, Dict[str, str]] = {}
	moved = 0
	for activity_type, activity_id, fmt, path in unknown_files:
		print(f"Inspecting activity {activity_id} ({activity_type}) from {path}")
		coords = _extract_latlng_from_file(path, fmt)
		if coords is None:
			print(f"Could not parse coordinates for activity {activity_id}; skipping.")
			continue
		lat, lng = coords
		loc = _resolve_location_from_coords(lat, lng, args, geocode_cache)
		if not loc:
			if not args.geocode:
				print(f"Coordinates found but --geocode not enabled; run with --geocode to resolve activity {activity_id}.")
			else:
				print(f"Reverse geocode did not yield a city/state/country for activity {activity_id}; skipping.")
			continue
		target = build_download_path(args.download_dir, activity_type, loc, activity_id, fmt)
		if os.path.abspath(path) == os.path.abspath(target):
			print(f"Activity {activity_id} already in final location: {target}")
			continue
		os.makedirs(os.path.dirname(target), exist_ok=True)
		try:
			os.replace(path, target)
		except OSError as e:
			print(f"Failed to move {path} -> {target}: {e}")
			continue
		print(f"Moved {activity_id} -> {target}")
		moved += 1
	print(f"Rescan complete: moved {moved} files.")
	return 0


def _resolve_location_from_coords(lat: float, lng: float, args, geocode_cache: Dict[str, Dict[str, str]]) -> Optional[Dict[str, str]]:
	"""Return a location dict for the given coordinates, using cache and optional geocoding."""
	if not args.geocode:
		return None
	# Skip invalid/placeholder coordinates (privacy or treadmill may result in 0,0)
	if abs(lat) < 1e-6 and abs(lng) < 1e-6:
		return None
	key = _coords_cache_key(lat, lng)
	if key not in geocode_cache:
		geo = reverse_geocode_nominatim(lat, lng)
		geocode_cache[key] = geo
		if args.geocode_sleep_ms:
			time.sleep(max(0, int(args.geocode_sleep_ms)) / 1000.0)
	geo = geocode_cache[key]
	if geo["city"] or geo["state"] or geo["country"]:
		return geo
	return None


def _coords_cache_key(lat: float, lng: float) -> str:
	return f"{lat:.3f},{lng:.3f}"


def _extract_latlng_from_file(path: str, fmt: str) -> Optional[Tuple[float, float]]:
	fmt = fmt.lower()
	if fmt in ("gpx", "tcx"):
		return _extract_latlng_from_xml(path, fmt)
	if fmt == "original":
		return _extract_latlng_from_original_zip(path)
	return None


def _extract_latlng_from_xml(path: str, fmt: str) -> Optional[Tuple[float, float]]:
	try:
		tree = ET.parse(path)
	except (ET.ParseError, OSError):
		return None
	root = tree.getroot()
	return _extract_latlng_from_root(root, fmt)


def _extract_latlng_from_stream(stream, fmt: str) -> Optional[Tuple[float, float]]:
	try:
		tree = ET.parse(stream)
	except (ET.ParseError, OSError):
		return None
	root = tree.getroot()
	return _extract_latlng_from_root(root, fmt)


def _extract_latlng_from_root(root: ET.Element, fmt: str) -> Optional[Tuple[float, float]]:
	fmt = fmt.lower()
	if fmt == "gpx":
		return _find_first_gpx_latlng(root)
	if fmt == "tcx":
		return _find_first_tcx_latlng(root)
	return None


def _find_first_gpx_latlng(root: ET.Element) -> Optional[Tuple[float, float]]:
	for elem in root.iter():
		tag = elem.tag.lower()
		if tag.endswith("trkpt"):
			lat = elem.attrib.get("lat")
			lon = elem.attrib.get("lon")
			if lat and lon:
				try:
					return float(lat), float(lon)
				except (ValueError, TypeError):
					continue
	return None


def _find_first_tcx_latlng(root: ET.Element) -> Optional[Tuple[float, float]]:
	for elem in root.iter():
		tag = elem.tag.lower()
		if tag.endswith("trackpoint"):
			lat_text = elem.findtext(".//{*}LatitudeDegrees")
			lon_text = elem.findtext(".//{*}LongitudeDegrees")
			if lat_text and lon_text:
				try:
					return float(lat_text), float(lon_text)
				except (ValueError, TypeError):
					continue
	return None

def _extract_latlng_from_original_zip(path: str) -> Optional[Tuple[float, float]]:
	try:
		with zipfile.ZipFile(path) as archive:
			for name in archive.namelist():
				if name.endswith("/"):
					continue
				lower = name.lower()
				if lower.endswith(".gpx"):
					with archive.open(name) as fp:
						coords = _extract_latlng_from_stream(fp, "gpx")
						if coords:
							return coords
				if lower.endswith(".tcx"):
					with archive.open(name) as fp:
						coords = _extract_latlng_from_stream(fp, "tcx")
						if coords:
							return coords
		return None
	except (zipfile.BadZipFile, OSError):
		return None


def scan_downloaded_activity_ids(base_dir: str) -> Set[int]:
	"""Scan the download directory for existing activity exports."""
	ids: Set[int] = set()
	if not os.path.isdir(base_dir):
		return ids
	for root, _, files in os.walk(base_dir):
		for file in files:
			name, _ = os.path.splitext(file)
			try:
				ids.add(int(name))
			except ValueError:
				continue
	return ids


def find_unknown_location_files(base_dir: str) -> List[Tuple[str, int, str, str]]:
	"""Return (activity_type, activity_id, fmt, path) for files under unknown locations."""
	results: List[Tuple[str, int, str, str]] = []
	if not os.path.isdir(base_dir):
		return results
	for root, _, files in os.walk(base_dir):
		rel = os.path.relpath(root, base_dir)
		if rel == ".":
			continue
		parts = rel.split(os.sep)
		if len(parts) < 4:
			continue
		activity_type = parts[0]
		loc_parts = parts[1:4]
		if not any(part.startswith("unknown_") for part in loc_parts):
			continue
		for file in files:
			name, ext = os.path.splitext(file)
			fmt = EXTENSION_TO_FORMAT.get(ext.lower())
			if not fmt:
				continue
			try:
				activity_id = int(name)
			except ValueError:
				continue
			results.append((activity_type, activity_id, fmt, os.path.join(root, file)))
	return results


def find_downloaded_gpx_files(base_dir: str) -> List[str]:
	"""Return every GPX file path under the download directory."""
	results: List[str] = []
	if not os.path.isdir(base_dir):
		return results
	for root, _, files in os.walk(base_dir):
		for file in files:
			if file.lower().endswith(".gpx"):
				results.append(os.path.join(root, file))
	return results




def run_retro_convert_tcx(args) -> int:
	"""Convert existing GPX downloads to TCX format under a target directory."""
	source_dir = args.download_dir
	if not os.path.isdir(source_dir):
		print(f"Download directory {source_dir} does not exist; nothing to convert.")
		return 1
	target_dir = args.retro_convert_tcx_dir or os.path.join(SCRIPT_ROOT, 'tcx_converted')
	gpx_files = find_downloaded_gpx_files(source_dir)
	if not gpx_files:
		print(f"No GPX files found under {source_dir}; nothing to convert.")
		return 0
	converted = 0
	skipped = 0
	for path in gpx_files:
		rel = os.path.relpath(path, source_dir)
		activity_type = rel.split(os.sep)[0].replace('_',' ') if os.sep in rel else ''
		# Walk-only guard: skip any activity type not 'Walk' with explicit message.
		if activity_type.lower() != 'walk':
			print(f"Skipping {path}: activity type '{activity_type}' not supported for --retro-convert-fit (walk-only; no implementation/testing).")
			continue
		activity_name = None
		try:
			root = ET.parse(path).getroot()
			name_el = root.find('.//{*}name')
			if name_el is not None and name_el.text:
				activity_name = name_el.text
		except ET.ParseError:
			pass
		dest = os.path.join(target_dir, os.path.splitext(rel)[0] + '.tcx')
		if os.path.exists(dest):
			skipped += 1
			continue
	# TCX conversion removed.

def run_retro_convert_fit(args) -> int:
	"""Convert existing GPX downloads to FIT format under a target directory.

	Generates a VERY minimal FIT file (not all fields). This aims for Garmin classification; if Garmin rejects, additional messages/fields may be required.
	"""
	source_dir = args.download_dir
	if not os.path.isdir(source_dir):
		print(f"Download directory {source_dir} does not exist; nothing to convert.")
		return 1
	target_dir = args.retro_convert_fit_dir or os.path.join(SCRIPT_ROOT, 'fit_converted')
	gpx_files = find_downloaded_gpx_files(source_dir)
	if not gpx_files:
		print(f"No GPX files found under {source_dir}; nothing to convert.")
		return 0
	converted = 0
	skipped = 0
	for path in gpx_files:
		rel = os.path.relpath(path, source_dir)
		activity_type = rel.split(os.sep)[0].replace('_',' ') if os.sep in rel else ''
		activity_name = None
		try:
			root = ET.parse(path).getroot()
			name_el = root.find('.//{*}name')
			if name_el is not None and name_el.text:
				activity_name = name_el.text
		except ET.ParseError:
			pass
		dest = os.path.join(target_dir, os.path.splitext(rel)[0] + '.fit')
		if os.path.exists(dest):
			skipped += 1
			continue
		ok = convert_gpx_file_to_fit(
			path, dest, activity_type, activity_name,
			prefer_generic_with_subsport=True,
			assume_naive_offset=getattr(args, 'assume_naive_offset', None),
			force_tz_offset_for_all=getattr(args, 'force_tz_offset_for_all', False),
			local_tz_name=getattr(args, 'local_tz', None),
			force_local_tz_for_all=getattr(args, 'force_local_tz_for_all', False),
		)
		if ok:
			converted += 1
		else:
			print(f"Failed to convert {path}")
	print(f"Retro FIT conversion complete: converted {converted}, skipped {skipped} existing.")
	print(f"FIT output tree written to {target_dir}")
	return 0

DOWNLOAD_EXTENSIONS = {
	"gpx": "gpx",
	"tcx": "tcx",
	"original": "zip",
}

EXTENSION_TO_FORMAT = {
	".gpx": "gpx",
	".tcx": "tcx",
	".zip": "original",
}


class StravaAuthError(Exception):
	pass


def load_credentials() -> Dict[str, str]:
	"""Load Strava credentials from env vars or strava_secrets.json.

	Returns a dict with keys: client_id, client_secret, refresh_token
	Raises StravaAuthError if not found.
	"""
	env = {
		"client_id": os.environ.get("STRAVA_CLIENT_ID"),
		"client_secret": os.environ.get("STRAVA_CLIENT_SECRET"),
		"refresh_token": os.environ.get("STRAVA_REFRESH_TOKEN"),
	}
	if all(env.values()):
		return env  # type: ignore

	# Try local JSON file next to this script
	secrets_path = os.path.join(os.path.dirname(__file__), "strava_secrets.json")
	if os.path.exists(secrets_path):
		try:
			with open(secrets_path, "r", encoding="utf-8") as f:
				data = json.load(f)
			required = {"client_id", "client_secret", "refresh_token"}
			if required.issubset(data):
				return {
					"client_id": str(data["client_id"]),
					"client_secret": str(data["client_secret"]),
					"refresh_token": str(data["refresh_token"]),
				}
		except Exception as e:
			raise StravaAuthError(f"Failed to read {secrets_path}: {e}")

	raise StravaAuthError(
		"Missing Strava credentials. Set STRAVA_CLIENT_ID, STRAVA_CLIENT_SECRET, STRAVA_REFRESH_TOKEN "
		"as environment variables, or create strava_secrets.json next to this script."
	)


def get_access_token(client_id: str, client_secret: str, refresh_token: str) -> str:
	"""Exchange a refresh token for a short-lived access token."""
	payload = {
		"client_id": client_id,
		"client_secret": client_secret,
		"grant_type": "refresh_token",
		"refresh_token": refresh_token,
	}
	resp = requests.post(STRAVA_TOKEN_URL, data=payload, timeout=30)
	if resp.status_code != 200:
		raise StravaAuthError(
			f"Token request failed: {resp.status_code} {resp.text[:300]}"
		)
	data = resp.json()
	access_token = data.get("access_token")
	if not access_token:
		raise StravaAuthError("No access_token in token response")
	return access_token


def build_auth_url(client_id: str, *, redirect_uri: str, scope: str = "read,activity:read_all", approval_prompt: str = "force") -> str:
	"""Construct the Strava OAuth authorization URL for the user to approve.

	scope options typically include 'read', 'activity:read', 'activity:read_all'.
	"""
	from urllib.parse import urlencode

	qs = urlencode(
		{
			"client_id": client_id,
			"response_type": "code",
			"redirect_uri": redirect_uri,
			"approval_prompt": approval_prompt,
			"scope": scope,
		}
	)
	return f"{STRAVA_AUTH_URL}?{qs}"


def exchange_auth_code(client_id: str, client_secret: str, code: str) -> Dict[str, str]:
	"""Exchange an OAuth authorization code for access and refresh tokens.

	Returns a dict including access_token, refresh_token, expires_at, and athlete id.
	"""
	payload = {
		"client_id": client_id,
		"client_secret": client_secret,
		"code": code,
		"grant_type": "authorization_code",
	}
	resp = requests.post(STRAVA_TOKEN_URL, data=payload, timeout=30)
	if resp.status_code != 200:
		raise StravaAuthError(
			f"Auth code exchange failed: {resp.status_code} {resp.text[:300]}"
		)
	data = resp.json()
	result = {
		"access_token": data.get("access_token"),
		"refresh_token": data.get("refresh_token"),
		"expires_at": str(data.get("expires_at")),
		"athlete_id": str((data.get("athlete") or {}).get("id", "")),
		"scope": ",".join(data.get("scope", [])) if isinstance(data.get("scope"), list) else str(data.get("scope", "")),
	}
	if not result["refresh_token"]:
		raise StravaAuthError("No refresh_token returned; ensure you used an authorization code, not an access token.")
	return result


def write_secrets_json(path: str, client_id: str, client_secret: str, refresh_token: str) -> None:
	payload = {
		"client_id": client_id,
		"client_secret": client_secret,
		"refresh_token": refresh_token,
	}
	with open(path, "w", encoding="utf-8") as f:
		json.dump(payload, f, indent=2)


def fetch_activity_page(
	access_token: str,
	*,
	page: int,
	per_page: int = 200,
	after: Optional[int] = None,
	before: Optional[int] = None,
) -> List[Dict]:
	"""Retrieve a single activities page with built-in rate-limit handling."""
	headers = {"Authorization": f"Bearer {access_token}"}
	params = {"page": page, "per_page": per_page}
	if after is not None:
		params["after"] = after
	if before is not None:
		params["before"] = before

	while True:
		resp = requests.get(STRAVA_ACTIVITIES_URL, headers=headers, params=params, timeout=60)
		if resp.status_code == 429:
			print(f"Rate limited fetching activities page {page}, waiting for next slot...", flush=True)
			wait_to_next_quarter()
			continue
		resp.raise_for_status()
		return resp.json()


def fetch_activity_details(access_token: str, activity_id: int, *, timeout_s: int = 60, retries: int = 3, backoff_s: int = 5, progress_cb=None) -> Dict:
	"""Fetch detailed activity to access city/state/country fields.

	Note: This costs one API call per activity and is subject to Strava rate limits
	(e.g., 100 requests/15min, 1000/day by default).
	"""
	headers = {"Authorization": f"Bearer {access_token}"}
	url = STRAVA_ACTIVITY_DETAIL_URL_TMPL.format(id=activity_id)
	attempt = 0
	while True:
		attempt += 1
		try:
			resp = requests.get(url, headers=headers, params={"include_all_efforts": "false"}, timeout=timeout_s)
			if resp.status_code == 429:
				if progress_cb:
					progress_cb(f"Activity {activity_id}: rate limited, waiting for next slot...")
				wait_to_next_quarter()
				continue
			resp.raise_for_status()
			return resp.json()
		except requests.RequestException as e:
			if attempt >= retries:
				if progress_cb:
					progress_cb(f"Activity {activity_id}: failed after {attempt} attempts: {e}")
				raise
			if progress_cb:
				progress_cb(f"Activity {activity_id}: error '{e}', retrying in {backoff_s}s...")
			time.sleep(backoff_s)

def safe_folder_name(value: Optional[str], default: str) -> str:
	"""Sanitize a folder name segment derived from location strings."""
	if not value:
		return default
	segment = re.sub(r"[\\/:*?\"<>|]+", "", value).strip()
	segment = segment.replace(" ", "_")
	return segment or default


def download_activity_file(access_token: str, activity_id: int, fmt: str) -> bytes:
	"""Download a Strava activity export (GPX/TCX/original)."""
	fmt = fmt.lower()
	if fmt not in STRAVA_DOWNLOAD_URLS:
		raise ValueError(f"Unsupported download format: {fmt}")
	url = STRAVA_DOWNLOAD_URLS[fmt].format(id=activity_id)
	headers = {"Authorization": f"Bearer {access_token}"}
	resp = requests.get(url, headers=headers, timeout=60)
	resp.raise_for_status()
	return resp.content



def build_download_path(base_dir: str, activity_type: str, location: Dict[str, str], activity_id: int, fmt: str) -> str:
	"""Construct a path like base/activity_type/country/state/city/activity.format."""
	activity_segment = safe_folder_name(activity_type, "unknown_activity")
	country = safe_folder_name(location.get("country", ""), "unknown_country")
	state = safe_folder_name(location.get("state", ""), "unknown_state")
	city = safe_folder_name(location.get("city", ""), "unknown_city")
	ext = DOWNLOAD_EXTENSIONS.get(fmt.lower(), fmt.lower())
	folder = os.path.join(base_dir, activity_segment, country, state, city)
	filename = f"{activity_id}.{ext}"
	return os.path.join(folder, filename)


class SimpleProgress:
	def __init__(self, enabled: bool):
		self.enabled = enabled
		self.spinner = itertools.cycle('|/-\\')
		self.last = 0.0

	def update(self, processed: int, details_calls: int, details_ok: int, downloads: int = 0):
		if not self.enabled:
			return
		now = time.time()
		if now - self.last < 0.1:
			return
		line = (
			f"\rProcessed: {processed} | details: {details_calls} (ok {details_ok})"
			f" | downloads: {downloads} {next(self.spinner)}"
		)
		try:
			sys.stdout.write(line)
			sys.stdout.flush()
		except Exception:
			pass
		self.last = now

	def done(self):
		if not self.enabled:
			return
		try:
			sys.stdout.write("\n")
			sys.stdout.flush()
		except Exception:
			pass


def fmt_duration(seconds: Optional[int]) -> str:
	if not seconds and seconds != 0:
		return ""
	seconds = int(seconds)
	h, rem = divmod(seconds, 3600)
	m, s = divmod(rem, 60)
	if h:
		return f"{h:d}:{m:02d}:{s:02d}"
	return f"{m:d}:{s:02d}"


def to_local_date_str(iso: Optional[str]) -> str:
	if not iso:
		return ""
	try:
		# start_date_local is ISO8601 with Z or offset
		return str(iso)
	except Exception:
		return str(iso)


def get_city_state_country(act: Dict) -> Dict[str, str]:
	"""Extract location fields (city/state/country) as strings, empty if missing."""
	def norm(v):
		return str(v).strip() if v is not None else ""

	return {
		"city": norm(act.get("location_city")),
		"state": norm(act.get("location_state")),
		"country": norm(act.get("location_country")),
	}


def _round_coord(value: float, precision: int = 3) -> float:
	"""Round coordinate to reduce geocode cache fragmentation (~100m at 3 decimals)."""
	return round(float(value), precision)


def _parse_iso_datetime(value: Optional[str]) -> Optional[dt.datetime]:
	if not value:
		return None
	try:
		if value.endswith("Z"):
			value = value[:-1] + "+00:00"
		return dt.datetime.fromisoformat(value)
	except ValueError:
		return None


def fetch_activity_streams(access_token: str, activity_id: int, *, timeout_s: int = 30) -> Dict[str, Dict]:
	"""Fetch activity streams keyed by type for GPX generation."""
	headers = {"Authorization": f"Bearer {access_token}"}
	url = STRAVA_ACTIVITY_STREAMS_URL_TMPL.format(id=activity_id)
	params = {"keys": ",".join(["latlng", "altitude", "time"]), "key_by_type": "true"}
	while True:
		resp = requests.get(url, headers=headers, params=params, timeout=timeout_s)
		if resp.status_code == 429:
			print(f"Rate limited fetching streams for activity {activity_id}, waiting for next slot...", flush=True)
			wait_to_next_quarter()
			continue
		resp.raise_for_status()
		return resp.json() or {}


def build_gpx_from_streams(activity: Dict, streams: Dict[str, Dict]) -> bytes:
	"""Construct a GPX blob from Strava activity streams.

	Timestamp handling:
	Strava provides both start_date (UTC) and start_date_local (local tz). Previously we
	used start_date_local which produced naive ISO strings (no timezone offset) in the
	GPX <time> elements. Later FIT conversion interpreted those as UTC, shifting times.
	We now always anchor to start_date (UTC) and emit explicit 'Z' suffixed timestamps.
	If start_date is missing, we fall back to parsing start_date_local; if that is naive
	we still treat it as UTC to preserve previous behavior but mark with 'Z'.
	"""
	latlng = streams.get("latlng", {}).get("data") or []
	if not latlng:
		raise ValueError("No location stream available to build GPX.")
	time_data = streams.get("time", {}).get("data") or []
	altitude_data = streams.get("altitude", {}).get("data") or []
	track = ET.Element("gpx", version="1.1", creator="strava.py", xmlns="http://www.topografix.com/GPX/1/1")
	trk = ET.SubElement(track, "trk")
	name = ET.SubElement(trk, "name")
	name.text = activity.get("name") or f"Activity {activity.get('id')}"
	trkseg = ET.SubElement(trk, "trkseg")
	# Prefer UTC start
	start_iso_utc = activity.get("start_date") or ""
	start_dt = _parse_iso_datetime(start_iso_utc)
	if start_dt and start_dt.tzinfo is None:
		# Assume UTC if Strava string lost its offset
		start_dt = start_dt.replace(tzinfo=dt.timezone.utc)
	if not start_dt:
		# Fallback to local date; if naive treat as UTC (best-effort)
		start_iso_local = activity.get("start_date_local") or ""
		start_dt = _parse_iso_datetime(start_iso_local)
		if start_dt and start_dt.tzinfo is None:
			start_dt = start_dt.replace(tzinfo=dt.timezone.utc)
	for idx, point in enumerate(latlng):
		if not point or len(point) != 2:
			continue
		trkpt = ET.SubElement(trkseg, "trkpt", lat=str(point[0]), lon=str(point[1]))
		if altitude_data and idx < len(altitude_data):
			ele = ET.SubElement(trkpt, "ele")
			ele.text = str(altitude_data[idx])
		if time_data and start_dt:
			seconds = float(time_data[idx]) if idx < len(time_data) else 0.0
			pt_time = start_dt + dt.timedelta(seconds=seconds)
			# Ensure UTC and 'Z' suffix
			if pt_time.tzinfo is None:
				pt_time = pt_time.replace(tzinfo=dt.timezone.utc)
			pt_time_utc = pt_time.astimezone(dt.timezone.utc).replace(microsecond=0)
			time_el = ET.SubElement(trkpt, "time")
			time_el.text = pt_time_utc.isoformat().replace('+00:00', 'Z')
	return ET.tostring(track, encoding="utf-8", xml_declaration=True)


def override_gpx_activity_type(content: bytes, activity_type: Optional[str]) -> bytes:
	"""Set the GPX <metadata><type> value to the given activity type."""
	if not activity_type:
		return content
	try:
		orig_root = ET.fromstring(content)
	except ET.ParseError:
		return content
	# Determine namespace handling. If original has a default namespace, rebuild a clean tree to avoid prefixed tags.
	if orig_root.tag.startswith("{"):
		ns_uri = orig_root.tag[1:].split("}", 1)[0]
		# Build new root with default namespace again
		new_root = ET.Element("gpx", version=orig_root.attrib.get("version", "1.1"), creator=orig_root.attrib.get("creator", "strava.py"), xmlns=ns_uri)
		# Copy existing children except existing metadata (we'll rebuild)
		existing_metadata = None
		for child in list(orig_root):
			local_tag = child.tag.split("}", 1)[1] if child.tag.startswith("{") else child.tag
			if local_tag == "metadata":
				existing_metadata = child
				continue
			# Append shallow copy (keep subtree)
			new_root.append(child)
		root = new_root
	else:
		root = orig_root
	# Find or create metadata element as first child
	metadata = root.find("metadata")
	if metadata is None:
		metadata = ET.Element("metadata")
		root.insert(0, metadata)
	# Mapping to Garmin lowercase activity keywords where possible
	garmin_map = {
		"Hike": "hiking",
		"Walk": "walking",
		"Run": "running",
		"Swim": "swimming",
		"Ride": "cycling",
		"Pickleball": "pickleball",
		"Tennis": "tennis",
		"Rock Climb": "climbing",
		"Volleyball": "other",  # Garmin may not have specific volleyball keyword; treat as other
	}
	mapped = garmin_map.get(activity_type, activity_type.lower())
	# Ensure <metadata><type>
	type_elem = metadata.find("type")
	if type_elem is None:
		type_elem = ET.SubElement(metadata, "type")
	type_elem.text = mapped
	# Also set <trk><type> to mapped (Garmin places it there)
	trk = root.find("trk")
	if trk is not None:
		trk_type = trk.find("type")
		if trk_type is None:
			trk_type = ET.SubElement(trk, "type")
		trk_type.text = mapped
	# Optionally add <time> if missing and we can infer start time from first <trkpt><time>
	time_elem = metadata.find("time")
	if time_elem is None:
		first_time = None
		for trkpt in root.findall('.//trkpt'):
			pt_time = trkpt.find('time')
			if pt_time is not None and pt_time.text:
				first_time = pt_time.text
				break
		if first_time:
			time_elem = ET.SubElement(metadata, 'time')
			time_elem.text = first_time
	return ET.tostring(root, encoding="utf-8", xml_declaration=True)

def convert_gpx_to_tcx(gpx_bytes: bytes, *, activity_type: str, activity_name: Optional[str] = None) -> bytes:
	"""Convert a GPX blob into a minimal TCX document for Garmin import.

	The TCX schema (Training Center XML) requires Activities->Activity(Sport)->Lap->Track->Trackpoint.
	We'll derive Sport from a mapping, use the first timestamp as Lap StartTime, and copy trackpoints with Time/Position/Altitude.
	Adds computed TotalTimeSeconds, DistanceMeters, MaximumSpeed from sequential points.
	"""
	# Map Strava/GPX activity type to TCX Sport enumeration
	sport_map = {
		"hike": "Walking",  # Map hike -> Walking to encourage correct classification; Garmin may infer Hiking specifics
		"walking": "Walking",
		"walk": "Walking",
		"run": "Running",
		"running": "Running",
		"ride": "Biking",
		"cycling": "Biking",
		"swim": "Other",  # Could choose Swimming if desired
		"pickleball": "Other",
		"tennis": "Other",
		"climbing": "Other",
		"rock climb": "Other",
		"volleyball": "Other",
	}
	try:
		gpx_root = ET.fromstring(gpx_bytes)
	except ET.ParseError:
		return b""
	# Gather trackpoints
	trkpts: List[Tuple[str, str, Optional[str], Optional[str]]] = []
	for trkpt in gpx_root.findall('.//{*}trkpt'):
		lat = trkpt.attrib.get('lat')
		lon = trkpt.attrib.get('lon')
		pt_time_el = trkpt.find('{*}time')
		pt_ele_el = trkpt.find('{*}ele')
		if not lat or not lon:
			continue
		pt_time = pt_time_el.text if pt_time_el is not None else None
		trkpts.append((lat, lon, pt_time, pt_ele_el.text if pt_ele_el is not None else None))
	if not trkpts:
		return b""  # Nothing to convert
	# Determine start time
	start_time: Optional[str] = None
	for _, _, t, _ in trkpts:
		if t:
			start_time = t
			break
	if not start_time:
		start_time = dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
	# Normalize timestamps to Z-form (Garmin tolerant of +00:00 but standard TCX sample uses Z)
	def _norm_ts(ts: str) -> str:
		if ts.endswith('+00:00'):
			return ts[:-6] + 'Z'
		return ts
	start_time = _norm_ts(start_time)
	# Compute total time, distance, max speed
	def _parse_iso(ts: str) -> Optional[dt.datetime]:
		try:
			if ts.endswith('Z'):
				return dt.datetime.fromisoformat(ts[:-1] + '+00:00')
			return dt.datetime.fromisoformat(ts)
		except Exception:
			return None
	def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
		# meters
		from math import radians, sin, cos, asin, sqrt
		R = 6371000.0
		dlat = radians(lat2 - lat1)
		dlon = radians(lon2 - lon1)
		a = sin(dlat/2)**2 + cos(radians(lat1))*cos(radians(lat2))*sin(dlon/2)**2
		c = 2 * asin(sqrt(a))
		return R * c
	total_distance_m = 0.0
	max_speed_mps = 0.0
	last_time_dt: Optional[dt.datetime] = None
	last_lat: Optional[float] = None
	last_lon: Optional[float] = None
	for lat, lon, t, _ in trkpts:
		try:
			lat_f = float(lat); lon_f = float(lon)
		except ValueError:
			continue
		cur_time_dt = _parse_iso(t) if t else None
		if last_lat is not None and last_lon is not None and cur_time_dt and last_time_dt:
			d = _haversine(last_lat, last_lon, lat_f, lon_f)
			dt_seconds = (cur_time_dt - last_time_dt).total_seconds()
			if d > 0:
				total_distance_m += d
				if dt_seconds > 0:
					spd = d / dt_seconds
					if spd > max_speed_mps:
						max_speed_mps = spd
		last_lat, last_lon, last_time_dt = lat_f, lon_f, cur_time_dt
	end_time_dt = last_time_dt or _parse_iso(start_time)
	start_time_dt = _parse_iso(start_time) or end_time_dt
	total_time_seconds = 0
	if start_time_dt and end_time_dt:
		span = (end_time_dt - start_time_dt).total_seconds()
		if span > 0:
			total_time_seconds = int(span)
	# Normalize activity type for mapping
	atype_norm = activity_type.lower()
	sport = sport_map.get(atype_norm, "Other")
	# Build TCX XML
	ns_tcx = "http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2"
	xsi = "http://www.w3.org/2001/XMLSchema-instance"
	TCX = ET.Element('TrainingCenterDatabase', {
		'xmlns': ns_tcx,
		'xmlns:xsi': xsi,
		'xsi:schemaLocation': f"{ns_tcx} {ns_tcx.replace('/v2', 'v2.xsd')}",
	})
	activities = ET.SubElement(TCX, 'Activities')
	activity = ET.SubElement(activities, 'Activity', Sport=sport)
	activity_id = ET.SubElement(activity, 'Id')
	activity_id.text = start_time
	lap = ET.SubElement(activity, 'Lap', StartTime=start_time)
	total_time = ET.SubElement(lap, 'TotalTimeSeconds'); total_time.text = str(total_time_seconds)
	distance = ET.SubElement(lap, 'DistanceMeters'); distance.text = f"{total_distance_m:.1f}"
	max_speed = ET.SubElement(lap, 'MaximumSpeed'); max_speed.text = f"{max_speed_mps:.2f}"
	calories = ET.SubElement(lap, 'Calories'); calories.text = '0'  # placeholder; Strava calorie export requires API detail
	trigger = ET.SubElement(lap, 'TriggerMethod'); trigger.text = 'Manual'
	intensity = ET.SubElement(lap, 'Intensity'); intensity.text = 'Active'
	track = ET.SubElement(lap, 'Track')
	for lat, lon, t, ele in trkpts:
		trackpt = ET.SubElement(track, 'Trackpoint')
		if t:
			when = ET.SubElement(trackpt, 'Time'); when.text = _norm_ts(t)
		pos = ET.SubElement(trackpt, 'Position')
		lat_el = ET.SubElement(pos, 'LatitudeDegrees'); lat_el.text = lat
		lon_el = ET.SubElement(pos, 'LongitudeDegrees'); lon_el.text = lon
		if ele:
			alt_el = ET.SubElement(trackpt, 'AltitudeMeters'); alt_el.text = ele
	# Activity Notes (name)
	if activity_name:
		notes = ET.SubElement(activity, 'Notes'); notes.text = activity_name
	return ET.tostring(TCX, encoding='utf-8', xml_declaration=True)


# ---------------- FIT Conversion (Minimal) -----------------

def _fit_crc16(data: bytes) -> int:
	"""Compute FIT CRC16 using official Dynastream nibble-table algorithm.

	Reference (FIT SDK C):
	static const uint16_t crc_table[16] = {
	  0x0000, 0xCC01, 0xD801, 0x1400,
	  0xF001, 0x3C00, 0x2800, 0xE401,
	  0xA001, 0x6C00, 0x7800, 0xB401,
	  0x5000, 0x9C01, 0x8801, 0x4400
	};
	uint16_t CRC_Get16(uint16_t crc, uint8_t byte) {
	  uint16_t tmp;
	  tmp = crc_table[crc & 0xF];
	  crc = (crc >> 4) & 0x0FFF;
	  crc ^= tmp ^ crc_table[byte & 0xF];
	  tmp = crc_table[crc & 0xF];
	  crc = (crc >> 4) & 0x0FFF;
	  crc ^= tmp ^ crc_table[(byte >> 4) & 0xF];
	  return crc;
	}
	We iterate CRC_Get16 for each byte; initial crc = 0.
	"""
	crc_table = (
		0x0000, 0xCC01, 0xD801, 0x1400,
		0xF001, 0x3C00, 0x2800, 0xE401,
		0xA001, 0x6C00, 0x7800, 0xB401,
		0x5000, 0x9C01, 0x8801, 0x4400,
	)
	crc = 0
	for b in data:
		# Low nibble
		tmp = crc_table[crc & 0xF]
		crc = (crc >> 4) & 0x0FFF
		crc ^= tmp ^ crc_table[b & 0xF]
		# High nibble
		tmp = crc_table[crc & 0xF]
		crc = (crc >> 4) & 0x0FFF
		crc ^= tmp ^ crc_table[(b >> 4) & 0xF]
	return crc & 0xFFFF

def convert_gpx_to_fit(
	gpx_bytes: bytes,
	*,
	activity_type: str,
	activity_name: Optional[str] = None,
	include_sport_message: bool = False,
	prefer_generic_with_subsport: bool = False,
	assume_naive_offset: Optional[str] = None,
	force_tz_offset_for_all: bool = False,
	local_tz_name: Optional[str] = None,
	force_local_tz_for_all: bool = False,
) -> bytes:
	"""Convert GPX bytes to a FIT activity file.

	Message order (lean): file_id -> device_info -> event(start) -> record* -> event(stop) -> lap -> session -> activity
	Scaled integer fields per FIT profile (time*1000, distance*100, speed*1000).

	Key compatibility adjustments:
	- Use protocol version 0x10 and profile version 0x1C2 (matches modern devices) rather than legacy 0x0100 which some Garmin ingestion paths flag.
	- Correct CRC16 algorithm (previous implementation produced non-Garmin CRCs).
	- Provide explicit manufacturer (Garmin=1) and a non-zero product id (generic=1) to avoid 'unknown device' rejection.
	- If activity_name present, include it encoded as UTF-8 in a session "sport" extension (field 3 custom name) so Connect can display it.
	"""
	try:
		root = ET.fromstring(gpx_bytes)
	except ET.ParseError:
		return b""
	points: List[Tuple[float, float, Optional[str], Optional[float]]] = []
	for trkpt in root.findall('.//{*}trkpt'):
		lat_s = trkpt.attrib.get('lat'); lon_s = trkpt.attrib.get('lon')
		if not lat_s or not lon_s:
			continue
		try:
			lat_f = float(lat_s); lon_f = float(lon_s)
		except ValueError:
			continue
		alt_el = trkpt.find('{*}ele')
		alt = None
		if alt_el is not None and alt_el.text:
			try: alt = float(alt_el.text)
			except ValueError: alt = None
		time_el = trkpt.find('{*}time')
		pt_time = time_el.text if time_el is not None and time_el.text else None
		points.append((lat_f, lon_f, pt_time, alt))
	if not points:
		return b""
	FIT_EPOCH = dt.datetime(1989, 12, 31, tzinfo=dt.timezone.utc)

	# Determine timezone to apply to naive timestamps if needed
	def _parse_offset_to_tz(offset: str) -> Optional[dt.timezone]:
		try:
			offset = offset.strip()
			m = re.match(r"^([+-])(\d{2}):(\d{2})$", offset)
			if not m:
				return None
			sign = -1 if m.group(1) == '-' else 1
			h = int(m.group(2)); mm = int(m.group(3))
			return dt.timezone(sign * dt.timedelta(hours=h, minutes=mm))
		except Exception:
			return None

	# Choose timezone policy (priority: local_tz_name -> assume_naive_offset -> system local)
	local_tz_for_naive: dt.tzinfo
	if local_tz_name and ZoneInfo is not None:
		try:
			local_tz_for_naive = ZoneInfo(local_tz_name)
		except Exception:
			print(f"Warning: could not load timezone '{local_tz_name}'. Falling back to fixed/system offset; DST may be incorrect. Consider: pip install tzdata")
			_tz = _parse_offset_to_tz(assume_naive_offset) if assume_naive_offset else None
			local_tz_for_naive = _tz or (dt.datetime.now().astimezone().tzinfo or dt.timezone.utc)
	elif assume_naive_offset:
		_tz = _parse_offset_to_tz(assume_naive_offset)
		local_tz_for_naive = _tz or (dt.datetime.now().astimezone().tzinfo or dt.timezone.utc)
	else:
		local_tz_for_naive = dt.datetime.now().astimezone().tzinfo or dt.timezone.utc
	def parse_ts(ts: str) -> Optional[int]:
		"""Parse an ISO8601 timestamp from GPX.

		Behavior:
		- 'Z' suffix -> treated as UTC.
		- Explicit offset like +hh:mm is respected.
		- Naive timestamps (no tzinfo) are assumed to be LOCAL TIME on this machine,
		  then converted to UTC. This fixes legacy GPX that stored local wall time without
		  a timezone (previously misinterpreted as UTC causing 3–4h shifts).
		"""
		try:
			iso = ts.strip()
			if iso.endswith('Z'):
				iso = iso[:-1] + '+00:00'
			ts_dt = dt.datetime.fromisoformat(iso)
			# Determine strategy
			if force_local_tz_for_all and local_tz_name and ZoneInfo is not None:
				wall = ts_dt.replace(tzinfo=None)
				assumed = wall.replace(tzinfo=local_tz_for_naive)
				ts_dt = assumed.astimezone(dt.timezone.utc)
			elif force_tz_offset_for_all:
				wall = ts_dt.replace(tzinfo=None)
				assumed = wall.replace(tzinfo=local_tz_for_naive)
				ts_dt = assumed.astimezone(dt.timezone.utc)
			elif ts_dt.tzinfo is None:
				ts_dt = ts_dt.replace(tzinfo=local_tz_for_naive).astimezone(dt.timezone.utc)
			return int((ts_dt - FIT_EPOCH).total_seconds())
		except Exception:
			return None
	start_ts_raw = next((p[2] for p in points if p[2]), None)
	end_ts_raw = next((p[2] for p in reversed(points) if p[2]), None)
	start_ts = parse_ts(start_ts_raw) if start_ts_raw else 0
	end_ts = parse_ts(end_ts_raw) if end_ts_raw else start_ts
	total_time_s = max(0, (end_ts - start_ts))
	# Distance & speed metrics
	from math import radians, sin, cos, asin, sqrt
	def haversine(a_lat, a_lon, b_lat, b_lon):
		R = 6371000.0
		dlat = radians(b_lat - a_lat); dlon = radians(b_lon - a_lon)
		a = sin(dlat / 2)**2 + cos(radians(a_lat)) * cos(radians(b_lat)) * sin(dlon / 2)**2
		c = 2 * asin(sqrt(a))
		return R * c
	distance_m = 0.0
	prev = None
	for lat, lon, t, _ in points:
		cur_ts = parse_ts(t) if t else None
		if prev and cur_ts and prev[2]:
			d = haversine(prev[0], prev[1], lat, lon)
			if d > 0:
				distance_m += d
		prev = (lat, lon, cur_ts)
	# Ascend / descend metrics (meters) from altitude stream
	total_ascent_m = 0.0
	total_descent_m = 0.0
	prev_alt: Optional[float] = None
	for _lat, _lon, _t, alt in points:
		if alt is None:
			continue
		if prev_alt is not None:
			diff = alt - prev_alt
			if diff > 0:
				total_ascent_m += diff
			elif diff < 0:
				total_descent_m += -diff
		prev_alt = alt
	avg_speed_mps = (distance_m / total_time_s) if total_time_s > 0 else 0.0
	# Sport mapping (FIT profile Sport enum):
	# 0 generic, 1 running, 2 cycling, 3 transition, 4 fitness_equipment, 5 swimming,
	# 8 tennis, 14 hiking, 15 walking.
	atype = activity_type.lower()
	# Primary sport determination
	sport_enum = 0; subsport_enum = 0
	if atype in ('run','running'): sport_enum = 1
	elif atype in ('ride','cycling'): sport_enum = 2
	elif atype in ('swim','swimming'): sport_enum = 5
	elif atype in ('walk','walking'): sport_enum = 11
	elif atype in ('hike','hiking'): sport_enum = 14
	elif atype in ('tennis',): sport_enum = 8
	# Walking display strategy: if prefer_generic_with_subsport and walking detected, force sport=0 generic, subsport=36 walking.
	is_walking = atype in ('walk','walking')
	if is_walking and prefer_generic_with_subsport:
		sport_enum = 0
	subsport_enum = 36 if is_walking or sport_enum == 11 else 0
	# Helpers for scaling
	scale_time = lambda secs: int(secs * 1000 + 0.5)
	scale_distance = lambda m: int(m * 100 + 0.5)
	scale_speed = lambda spd: int(spd * 1000 + 0.5)
	SEMICIRCLE = 2147483648 / 180.0
	# Message builders
	body = bytearray()
	def def_header(local): return 0x40 | (local & 0x0F)
	def data_header(local): return local & 0x0F
	def def_message(local, global_num, fields, dev_fields=None):
		"""Build a definition message.

		If dev_fields provided, set developer flag (0x20) and append developer field count and triples (field_num,size,dev_data_index).
		"""
		header = 0x40 | (local & 0x0F)
		if dev_fields:
			header |= 0x20  # developer flag
		msg = bytearray([header, 0x00, 0x00])
		msg += global_num.to_bytes(2, 'little')
		msg.append(len(fields))
		for num, size, base in fields:
			msg += bytes([num, size, base])
		if dev_fields:
			msg.append(len(dev_fields))
			for fnum, fsize, findex in dev_fields:
				msg += bytes([fnum, fsize, findex])
		return msg
	# Minimal file: omit developer data and other optional messages.
	# file_id (local 0) -- restore correct definition (global 0)
	file_id_def = def_message(0, 0, [
		(0, 1, 0x00),  # type
		(1, 2, 0x84),  # manufacturer
		(2, 2, 0x84),  # product
		(3, 4, 0x86),  # serial_number
		(4, 4, 0x86),  # time_created
		(5, 2, 0x84),  # number (activity number)
	])
	body += file_id_def
	file_id_data = bytearray([data_header(0)])
	file_id_data += bytes([4])  # activity file
	file_id_data += (1).to_bytes(2, 'little')  # Garmin manufacturer
	file_id_data += (1).to_bytes(2, 'little')  # product (generic non-zero)
	file_id_data += (12345678).to_bytes(4, 'little')  # serial_number placeholder
	file_id_data += start_ts.to_bytes(4, 'little')    # time_created
	file_id_data += (1).to_bytes(2, 'little')        # activity number
	body += file_id_data
	# Optional sport message (global 12) to provide textual sport name for external tools.
	if include_sport_message:
		# Title-case mapping to align with reference FIT sample naming (e.g., 'Walk')
		name_map = {0:'Generic',1:'Run',2:'Ride',5:'Swim',8:'Tennis',11:'Walk',14:'Hike'}
		name_bytes = name_map.get(sport_enum,'activity').encode('utf-8')[:128]
		name_padded = name_bytes + b'\x00' * (128 - len(name_bytes))
		sport_def = def_message(7, 12, [
			(0,1,0x00),  # sport enum
			(1,1,0x00),  # subsport enum
			(3,128,0x07), # name (128 bytes standard)
		])
		body += sport_def
		sport_data = bytearray([data_header(7)])
		sport_data += bytes([sport_enum, subsport_enum])
		sport_data += name_padded
		body += sport_data
	# (Removed event messages for minimal file.)
	# record (local 3) add cumulative distance (field 5) & instantaneous speed (field 6)
	record_def = def_message(3, 20, [
		(253, 4, 0x86),  # timestamp
		(0, 4, 0x85),    # position_lat (sint32)
		(1, 4, 0x85),    # position_long (sint32)
		(2, 2, 0x84),    # altitude (uint16 scale=5 offset=500)
		(5, 4, 0x86),    # distance (uint32 cm scale=100)
		(6, 2, 0x84),    # speed (uint16 mm/s scale=1000)
	])
	body += record_def
	prev_lat = None; prev_lon = None; prev_ts = None
	cum_distance_m = 0.0
	max_inst_speed_mps = 0.0  # track maximum instantaneous segment speed for lap/session max_speed
	# Clamp helpers for field size safety
	def clamp_uint16(val: int) -> int:
		return 0 if val < 0 else (65535 if val > 65535 else val)
	def clamp_uint32(val: int) -> int:
		return 0 if val < 0 else (0xFFFFFFFF if val > 0xFFFFFFFF else val)
	for lat, lon, t, alt in points:
		cur_ts = parse_ts(t) if t else start_ts
		# Incremental distance
		if prev_lat is not None and prev_lon is not None and prev_ts is not None and cur_ts > prev_ts:
			seg_d = haversine(prev_lat, prev_lon, lat, lon)
			if seg_d > 0:
				cum_distance_m += seg_d
			seg_dt = cur_ts - prev_ts
			inst_speed = (seg_d / seg_dt) if seg_dt > 0 and seg_d > 0 else 0.0
		else:
			inst_speed = 0.0
		# Cap unrealistic instantaneous speed for walking/hiking to 3 m/s (use activity flag)
		if is_walking and inst_speed > 3.0:
			inst_speed = 3.0
		# Track max instantaneous speed
		if inst_speed > max_inst_speed_mps:
			max_inst_speed_mps = inst_speed
		data = bytearray([data_header(3)])
		data += cur_ts.to_bytes(4, 'little')
		lat_sc = int(lat * SEMICIRCLE); lon_sc = int(lon * SEMICIRCLE)
		data += lat_sc.to_bytes(4, 'little', signed=True)
		data += lon_sc.to_bytes(4, 'little', signed=True)
		alt_raw = int(((alt if alt is not None else 0) + 500) * 5)
		alt_raw = clamp_uint16(alt_raw)
		data += alt_raw.to_bytes(2, 'little')
		dist_raw = scale_distance(cum_distance_m)
		dist_raw = clamp_uint32(dist_raw)
		data += dist_raw.to_bytes(4, 'little')  # cumulative distance
		speed_raw = scale_speed(inst_speed)
		speed_raw = clamp_uint16(speed_raw)
		data += speed_raw.to_bytes(2, 'little')          # instantaneous speed
		body += data
		prev_lat, prev_lon, prev_ts = lat, lon, cur_ts
	# Events omitted for minimal file.
	# lap (local 4)
	start_lat, start_lon = points[0][0], points[0][1]
	end_lat, end_lon = points[-1][0], points[-1][1]
	# Reference-style lap definition expanded to include calories & avg/max speed in standard field positions.
	lap_def = def_message(4, 19, [
		(253, 4, 0x86),  # timestamp (end of lap)
		(2, 4, 0x86),    # start_time
		(3, 4, 0x85),    # start_position_lat
		(4, 4, 0x85),    # start_position_long
		(5, 4, 0x85),    # end_position_lat
		(6, 4, 0x85),    # end_position_long
		(7, 4, 0x86),    # total_elapsed_time (ms)
		(8, 4, 0x86),    # total_timer_time (ms)
		(9, 4, 0x86),    # total_distance (cm)
		(13, 2, 0x84),   # avg_speed (mm/s)
		(14, 2, 0x84),   # max_speed (mm/s)
		(21, 2, 0x84),   # total_ascent (m)
		(22, 2, 0x84),   # total_descent (m)
		(25, 1, 0x00),   # sport
		(26, 1, 0x00),   # subsport
		(254, 2, 0x84),  # message_index
	])
	body += lap_def
	lap_data = bytearray([data_header(4)])
	lap_data += end_ts.to_bytes(4, 'little')          # timestamp (end)
	lap_data += start_ts.to_bytes(4, 'little')        # start_time
	start_lat_sc = int(start_lat * SEMICIRCLE); start_lon_sc = int(start_lon * SEMICIRCLE)
	end_lat_sc = int(end_lat * SEMICIRCLE); end_lon_sc = int(end_lon * SEMICIRCLE)
	lap_data += start_lat_sc.to_bytes(4, 'little', signed=True)
	lap_data += start_lon_sc.to_bytes(4, 'little', signed=True)
	lap_data += end_lat_sc.to_bytes(4, 'little', signed=True)
	lap_data += end_lon_sc.to_bytes(4, 'little', signed=True)
	lap_data += scale_time(total_time_s).to_bytes(4, 'little')    # total_elapsed_time (7)
	lap_data += scale_time(total_time_s).to_bytes(4, 'little')    # total_timer_time (8)
	lap_data += scale_distance(distance_m).to_bytes(4, 'little')  # total_distance (9)
	# Compute avg & max speed (mm/s scaled) for lap: reuse overall avg & tracked max instantaneous
	# Track max instantaneous speed during record loop earlier (we'll add variable if missing).
	# avg_speed_mps already computed above.
	avg_speed_raw = scale_speed(avg_speed_mps)
	# Ensure max_inst_speed_mps obeys walking cap before scaling
	if is_walking and max_inst_speed_mps > 3.0:
		max_inst_speed_mps = 3.0
	max_speed_raw = scale_speed(max_inst_speed_mps if 'max_inst_speed_mps' in locals() else avg_speed_mps)
	avg_speed_raw = clamp_uint16(avg_speed_raw)
	max_speed_raw = clamp_uint16(max_speed_raw)
	lap_data += avg_speed_raw.to_bytes(2, 'little')    # avg_speed (13)
	lap_data += max_speed_raw.to_bytes(2, 'little')    # max_speed (14)
	# total_ascent / total_descent stored as whole meters
	lap_data += int(round(total_ascent_m)).to_bytes(2, 'little')  # total_ascent (21)
	lap_data += int(round(total_descent_m)).to_bytes(2, 'little') # total_descent (22)
	lap_data += bytes([sport_enum])   # sport (25)
	lap_data += bytes([subsport_enum])# subsport (26)
	lap_data += (0).to_bytes(2, 'little')  # message_index (254)
	# No developer fields appended (removed).
	body += lap_data
	# session (local 5)
	# Reorder definition so sport/subsport (and duplicates) appear after speed/ascent/descent and lap metadata.
	session_def = def_message(5, 18, [
			(253, 4, 0x86),  # timestamp
			(2, 4, 0x86),    # start_time
			(7, 4, 0x86),    # total_elapsed_time (ms)
			(8, 4, 0x86),    # total_timer_time (ms)
			(9, 4, 0x86),    # total_distance (cm)
			(13, 2, 0x84),   # avg_speed (mm/s)
			(14, 2, 0x84),   # max_speed (mm/s)
			(21, 2, 0x84),   # total_ascent (m)
			(22, 2, 0x84),   # total_descent (m)
			(3, 1, 0x00),    # first_lap_index (0)
			(4, 2, 0x84),    # num_laps (uint16)
			(5, 1, 0x00),    # sport (moved from 25)
			(6, 1, 0x00),    # subsport (moved from 26)
			(199, 1, 0x00),  # duplicate sport
			(200, 1, 0x00),  # duplicate subsport
			(254, 2, 0x84),  # message_index
		])
	body += session_def
	session_data = bytearray([data_header(5)])
	session_data += end_ts.to_bytes(4, 'little')             # timestamp
	session_data += start_ts.to_bytes(4, 'little')           # start_time
	session_data += scale_time(total_time_s).to_bytes(4, 'little')   # total_elapsed_time (7)
	session_data += scale_time(total_time_s).to_bytes(4, 'little')   # total_timer_time (8)
	session_data += scale_distance(distance_m).to_bytes(4, 'little') # total_distance (9)
	avg_speed_raw = scale_speed(avg_speed_mps)
	max_speed_raw = scale_speed(max_inst_speed_mps if 'max_inst_speed_mps' in locals() else avg_speed_mps)
	avg_speed_raw = clamp_uint16(avg_speed_raw)
	max_speed_raw = clamp_uint16(max_speed_raw)
	session_data += avg_speed_raw.to_bytes(2, 'little')             # avg_speed (13)
	session_data += max_speed_raw.to_bytes(2, 'little')             # max_speed (14)
	session_data += int(round(total_ascent_m)).to_bytes(2, 'little')  # total_ascent (21)
	session_data += int(round(total_descent_m)).to_bytes(2, 'little') # total_descent (22)
	session_data += bytes([0])                                      # first_lap_index (3)
	session_data += (1).to_bytes(2, 'little')                       # num_laps (4) - single lap
	def _session_sport_fields(is_walk: bool, sport_val: int, subsport_val: int):
		"""Return (primary_sport, primary_subsport, dup_sport, dup_subsport) for session.

		Walking: primary (11,0) duplicates (11,36). Others: mirror original.
		"""
		if is_walk:
			return 11, 0, 11, 36
		return sport_val, subsport_val, sport_val, subsport_val
	primary_sport, primary_subsport, dup_sport, dup_subsport = _session_sport_fields(is_walking, sport_enum, subsport_enum)
	session_data += bytes([primary_sport])                          # sport (5)
	session_data += bytes([primary_subsport])                       # subsport (6)
	session_data += bytes([dup_sport])                              # duplicate sport (199)
	session_data += bytes([dup_subsport])                           # duplicate subsport (200)
	session_data += (0).to_bytes(2, 'little')                       # message_index (254)
	# No developer sport_text appended.
	body += session_data
	# activity (local 6)
	activity_def = def_message(6, 34, [
		(253, 4, 0x86),  # timestamp
		(4, 4, 0x86),    # total_timer_time (ms)
		(5, 4, 0x86),    # local_timestamp
		(1, 2, 0x84),    # num_sessions
		(2, 1, 0x00),    # type (manual=0)
		(3, 2, 0x84),    # event_group (placeholder 0)
		(254, 2, 0x84),  # message_index
	])
	body += activity_def
	activity_data = bytearray([data_header(6)])
	activity_data += end_ts.to_bytes(4, 'little')  # timestamp (final time, UTC)
	activity_data += scale_time(total_time_s).to_bytes(4, 'little')  # total_timer_time
	# Compute local_timestamp as local wall time mapped to FIT epoch
	try:
		utc_dt = FIT_EPOCH + dt.timedelta(seconds=end_ts)
		local_dt = utc_dt.astimezone(local_tz_for_naive)
		local_ts = int((local_dt - FIT_EPOCH).total_seconds())
	except Exception:
		local_ts = end_ts
	activity_data += local_ts.to_bytes(4, 'little')  # local_timestamp (local wall time)
	activity_data += (1).to_bytes(2, 'little')     # num_sessions
	activity_data += bytes([0])                    # type manual
	activity_data += (0).to_bytes(2, 'little')     # event_group
	activity_data += (0).to_bytes(2, 'little')     # message_index
	body += activity_data
	# File header
	data_size = len(body)
	header = bytearray()
	header.append(14)          # header size
	header.append(0x10)        # protocol version 1.0
	# Use a more recent profile version (match working reference ~0x03C8 or close). Using 0x01C2 as safe middle ground.
	header += (0x03C8).to_bytes(2, 'little')  # profile version (968 decimal) newer for better tool compatibility
	header += data_size.to_bytes(4, 'little')
	header += b'.FIT'
	header += b'\x00\x00'    # placeholder CRC
	crc_header = _fit_crc16(bytes(header[:-2]))
	header[-2:] = crc_header.to_bytes(2, 'little')
	file_wo_crc = bytes(header) + bytes(body)
	file_crc = _fit_crc16(file_wo_crc)
	return file_wo_crc + file_crc.to_bytes(2, 'little')

def convert_gpx_file_to_fit(
	gpx_path: str,
	fit_path: str,
	activity_type: str,
	activity_name: Optional[str] = None,
	prefer_generic_with_subsport: bool = True,
	assume_naive_offset: Optional[str] = None,
	force_tz_offset_for_all: bool = False,
	local_tz_name: Optional[str] = None,
	force_local_tz_for_all: bool = False,
) -> bool:
	try:
		with open(gpx_path, 'rb') as f:
			data = f.read()
			fit = convert_gpx_to_fit(
				data,
				activity_type=activity_type,
				activity_name=activity_name,
				prefer_generic_with_subsport=prefer_generic_with_subsport,
				assume_naive_offset=assume_naive_offset,
				force_tz_offset_for_all=force_tz_offset_for_all,
				local_tz_name=local_tz_name,
				force_local_tz_for_all=force_local_tz_for_all,
			)
			if not fit:
				return False
			os.makedirs(os.path.dirname(fit_path), exist_ok=True)
			with open(fit_path, 'wb') as out:
				out.write(fit)
			return True
	except OSError:
		return False

def validate_fit(path: str) -> Dict[str, Any]:
	"""FIT validator & lightweight decoder.

	Checks header & file CRC, collects definitions, decodes a subset of data messages using definitions:
	- Timestamps (253)
	- Lat/Lon (0,1) scaled semicircles -> degrees
	- Altitude (2) uint16 scale=5 offset=500 -> meters
	- Scaled time/distance/speed (ms, cm, mm/s) for lap/session/activity
	Returns dict with keys: ok, error, protocol_version, profile_version, data_size, definitions, samples(list).
	"""
	info: Dict[str, Any] = {"ok": False, "error": None, "definitions": [], "samples": []}
	try:
		with open(path, 'rb') as f:
			data = f.read()
	except OSError as e:
		info["error"] = f"read failed: {e}"; return info
	if len(data) < 14:
		info["error"] = "file too short"; return info
	header_size = data[0]
	if header_size < 12 or len(data) < header_size + 2:
		info["error"] = "invalid header size"; return info
	protocol_ver = data[1]
	profile_ver = int.from_bytes(data[2:4], 'little')
	data_size = int.from_bytes(data[4:8], 'little')
	sig = data[8:12]
	if sig != b'.FIT':
		info["error"] = "missing .FIT signature"; return info
	if header_size >= 14:
		exp_header_crc = int.from_bytes(data[12:14], 'little')
		calc_header_crc = _fit_crc16(data[:12])
		if exp_header_crc != calc_header_crc:
			info["error"] = f"header CRC mismatch (exp {exp_header_crc:04x} got {calc_header_crc:04x})"; return info
	body = data[header_size:header_size+data_size]
	if len(body) != data_size:
		info["error"] = "data size mismatch"; return info
	file_crc_expected = int.from_bytes(data[header_size+data_size:header_size+data_size+2], 'little') if len(data) >= header_size+data_size+2 else None
	calc_file_crc = _fit_crc16(data[:header_size+data_size])
	if file_crc_expected is None or file_crc_expected != calc_file_crc:
		info["error"] = f"file CRC mismatch (exp {file_crc_expected} got {calc_file_crc})"; return info
	# Parse stream
	definitions: Dict[int, Dict[str, Any]] = {}
	# Minimal decoding: no sport metadata propagation or developer fields.
	i = 0
	while i < len(body):
		hdr = body[i]
		local_num = hdr & 0x0F
		is_def = (hdr & 0x40) == 0x40
		dev_flag = (hdr & 0x20) == 0x20
		if is_def:
			# Definition message layout (uncompressed):
			# header(1) | reserved(1) | architecture(1) | global_num(2) | field_count(1) | fields...
			# Constructed as: [hdr, reserved(0), arch(0)] + global_num(2 LE) + field_count + fields
			# Therefore:
			#  global_num starts at i+3 (LSB at i+3, MSB at i+4)
			#  field_count at i+5
			#  first field triple starts at i+6
			if i + 6 > len(body): break
			arch = body[i+2]  # architecture byte
			global_num = int.from_bytes(body[i+3:i+5], 'little')
			field_count = body[i+5]
			j = i + 6
			fields = []
			for _ in range(field_count):
				if j + 3 > len(body): break
				fn = body[j]; sz = body[j+1]; base = body[j+2]; j += 3
				fields.append((fn, sz, base))
			dev_fields = []
			if dev_flag and j < len(body):
				if j >= len(body):
					definitions[local_num] = {"global": global_num, "fields": fields, "dev_fields": dev_fields}
					info["definitions"].append({"local": local_num, "global": global_num, "fields": fields, "dev_fields": dev_fields})
					i = j; continue
				dev_count = body[j]; j += 1
				for _ in range(dev_count):
					if j + 3 > len(body): break
					dfn = body[j]; dsz = body[j+1]; dindex = body[j+2]; j += 3
					dev_fields.append((dfn, dsz, dindex))
			definitions[local_num] = {"global": global_num, "fields": fields, "dev_fields": dev_fields}
			info["definitions"].append({"local": local_num, "global": global_num, "fields": fields, "dev_fields": dev_fields})
			i = j
		else:
			# Data message decode using definition (if present)
			defn = definitions.get(local_num)
			if not defn:
				i += 1; continue
			cursor = i + 1
			decoded: Dict[str, Any] = {"local": local_num, "global": defn["global"]}
			for fn, sz, base in defn["fields"]:
				if cursor + sz > len(body): break
				raw = body[cursor:cursor+sz]
				cursor += sz
				val: Any = raw
				if sz == 1:
					val_int = raw[0]
				elif sz == 2:
					val_int = int.from_bytes(raw, 'little')
				elif sz == 4:
					val_int = int.from_bytes(raw, 'little', signed=(base == 0x85))
				else:
					val_int = int.from_bytes(raw, 'little')
				# Field-specific decode
				if fn == 253:  # timestamp seconds since FIT epoch
					val = val_int
				elif defn["global"] == 20 and fn in (0,1):  # record lat/lon semicircles
					val = (val_int / (2147483648/180.0))
				elif defn["global"] == 20 and fn == 2:  # altitude
					val = (val_int / 5.0) - 500.0
				elif defn["global"] in (18,19,34):
					# scaled fields for session/lap/activity (global 18,19,34)
					# NOTE: field 4 in session is num_laps (uint16) not a time; exclude from time scaling.
					if fn in (7,8) or (defn["global"] == 34 and fn == 4):  # times ms (activity uses field 4)
						val = val_int / 1000.0
					elif fn == 9:  # distance cm
						val = val_int / 100.0
					elif fn in (13,14):  # avg_speed / max_speed mm/s
						val = val_int / 1000.0
					else:
						val = val_int
				elif defn["global"] == 12:
					# Sport message: decode sport/subsport and name field
					if fn in (0,1):
						val = val_int
					elif fn == 3 and sz in (32,128):
						try:
							val = raw.split(b'\x00',1)[0].decode('utf-8','ignore')
						except Exception:
							val = raw
					else:
						val = val_int
				else:
					val = val_int
				decoded[f"f{fn}"] = val
				# Derive human-readable activity alias for session based on effective sport (prefer duplicate field 199 then primary 25)
				if defn["global"] == 18 and "activity" not in decoded:
					# Priority: f5 (new sport position) -> f199 (duplicate) -> f25 (legacy primary)
					effective_sport = None
					for key in ("f5","f199","f25"):
						val_s = decoded.get(key)
						if isinstance(val_s, int):
							effective_sport = val_s; break
					alias_map = {0: 'Generic', 1: 'Run', 2: 'Ride', 5: 'Swim', 8: 'Tennis', 11: 'Walk', 14: 'Hike'}
					if isinstance(effective_sport, int):
						decoded["activity"] = alias_map.get(effective_sport, 'Activity')
				# Propagate last sport name/popularity to session records for easier inspection
			# No developer field decoding (removed from minimal generator)
			# Removed sport/subsport canonical injection; rely on raw field numbers (f25/f199 and f26/f200).
			info["samples"].append(decoded)
			i = cursor
	info.update({"ok": True, "protocol_version": protocol_ver, "profile_version": profile_ver, "data_size": data_size})
	# Post-parse: ensure session messages have sport key; fallback from field numbers if absent.
	for sample in info.get("samples", []):
		if sample.get("global") == 18 and "sport" not in sample:
			# Try field 25 first
			val25 = sample.get("f25")
			if isinstance(val25, int):
				sample["sport"] = val25
			else:
				# Infer from subsport walking (36) -> walking sport (11)
				subs = sample.get("subsport") or sample.get("f26")
				if subs == 36:
					sample["sport"] = 11
				else:
					sample["sport"] = 0  # generic fallback
	return info


def print_session_columns(path: str):
	"""Convenience: print decoded session columns including effective sport (raw field) & activity alias.

	Sport display preference: f199 (duplicate sport) then f25. Subsport: f200 then f26.
	"""
	res = validate_fit(path)
	if not res.get("ok"):
		print(f"{path}: FAIL {res.get('error')}")
		return
	session_samples = [s for s in res.get("samples", []) if s.get("global") == 18]
	if not session_samples:
		print("No session messages found")
		return
	# Consolidate multiple session entries into one by combining earliest start_time and latest metrics.
	# Preference: use the last sample for totals (elapsed/timer/distance/ascent) but earliest start_time.
	earliest_start = None
	merged = None
	for s in session_samples:
		if earliest_start is None or (s.get("f2") is not None and s.get("f2") < earliest_start):
			earliest_start = s.get("f2")
		merged = s if merged is None else {**merged, **s}
	if merged is None:
		print("No session messages found")
		return
	merged["f2"] = earliest_start  # enforce earliest start_time
	# Columns focused on core metrics plus sport & sub sport (text)
	cols = ["timestamp","start_time","total_elapsed_time","total_timer_time","total_distance","total_ascent","average_speed","max_speed","f_sport","f_subsport"]
	# Map internal field keys to friendly names
	def get(sample, key):
		if key == "timestamp": return sample.get("f253")
		if key == "start_time": return sample.get("f2")
		if key == "total_elapsed_time": return sample.get("f7")
		if key == "total_timer_time": return sample.get("f8")
		if key == "total_distance": return sample.get("f9")
		if key == "total_ascent": return sample.get("f21")
		if key == "average_speed": return sample.get("f13")
		if key == "max_speed": return sample.get("f14")
		if key == "f_sport":
			val = None
			for k in ("f5","f199","f25"):
				if sample.get(k) is not None:
					val = sample.get(k); break
			map_enum = {0:'generic',1:'running',2:'cycling',5:'swimming',8:'tennis',11:'walking',14:'hiking'}
			return map_enum.get(val, str(val).lower() if val is not None else '')
		if key == "f_subsport":
			val = None
			for k in ("f6","f200","f26"):
				if sample.get(k) is not None:
					val = sample.get(k); break
			sub_map = {0:'generic',36:'walking'}
			return sub_map.get(val, str(val).lower() if val is not None else '')
		return sample.get(key)
	print("\t".join(cols))
	values = [str(get(merged,c) if get(merged,c) is not None else '') for c in cols]
	print("\t".join(values))
def reverse_geocode_nominatim(lat: float, lng: float, *, timeout_s: int = 15) -> Dict[str, str]:
	"""Reverse geocode using OpenStreetMap Nominatim.

	Returns dict with keys city, state, country (empty strings if unavailable).
	Respect rate limits when calling this in a loop.
	"""
	url = "https://nominatim.openstreetmap.org/reverse"
	params = {
		"lat": f"{lat:.6f}",
		"lon": f"{lng:.6f}",
		"format": "json",
		# Higher zoom provides better city-level results from OSM
		"zoom": 14,
		"addressdetails": 1,
	}
	headers = {
		"User-Agent": "strava-activities-script/1.0 (non-commercial)",
		"Accept-Language": "en",
	}
	try:
		resp = requests.get(url, params=params, headers=headers, timeout=timeout_s)
		resp.raise_for_status()
		data = resp.json() or {}
		addr = data.get("address") or {}
		# Try common OSM fields for locality name
		city = addr.get("city") or addr.get("town") or addr.get("village") or addr.get("hamlet") or ""
		state = addr.get("state") or ""
		country = addr.get("country") or ""
		return {"city": str(city), "state": str(state), "country": str(country)}
	except requests.RequestException:
		return {"city": "", "state": "", "country": ""}


def print_table(rows: List[Dict]):
	# Determine column widths
	headers = ["id", "type", "name", "date", "city", "state", "country", "distance_km", "moving_time"]
	col_widths = {h: len(h) for h in headers}
	for r in rows:
		col_widths["id"] = max(col_widths["id"], len(str(r.get("id", ""))))
		col_widths["type"] = max(col_widths["type"], len(str(r.get("type", ""))))
		col_widths["name"] = max(col_widths["name"], len(str(r.get("name", ""))))
		col_widths["date"] = max(col_widths["date"], len(str(r.get("date", ""))))
		col_widths["city"] = max(col_widths["city"], len(str(r.get("city", ""))))
		col_widths["state"] = max(col_widths["state"], len(str(r.get("state", ""))))
		col_widths["country"] = max(col_widths["country"], len(str(r.get("country", ""))))
		col_widths["distance_km"] = max(col_widths["distance_km"], len(str(r.get("distance_km", ""))))
		col_widths["moving_time"] = max(col_widths["moving_time"], len(str(r.get("moving_time", ""))))

	def fmt_row(r: Dict) -> str:
		return (
			f"{str(r.get('id','')).ljust(col_widths['id'])}  "
			f"{str(r.get('type','')).ljust(col_widths['type'])}  "
			f"{str(r.get('name','')).ljust(col_widths['name'])}  "
			f"{str(r.get('date','')).ljust(col_widths['date'])}  "
			f"{str(r.get('city','')).ljust(col_widths['city'])}  "
			f"{str(r.get('state','')).ljust(col_widths['state'])}  "
			f"{str(r.get('country','')).ljust(col_widths['country'])}  "
			f"{str(r.get('distance_km','')).rjust(col_widths['distance_km'])}  "
			f"{str(r.get('moving_time','')).rjust(col_widths['moving_time'])}"
		)

	header_line = (
		f"{ 'id'.ljust(col_widths['id']) }  "
		f"{ 'type'.ljust(col_widths['type']) }  "
		f"{ 'name'.ljust(col_widths['name']) }  "
		f"{ 'date'.ljust(col_widths['date']) }  "
		f"{ 'city'.ljust(col_widths['city']) }  "
		f"{ 'state'.ljust(col_widths['state']) }  "
		f"{ 'country'.ljust(col_widths['country']) }  "
		f"{ 'distance_km'.rjust(col_widths['distance_km']) }  "
		f"{ 'moving_time'.rjust(col_widths['moving_time']) }"
	)
	sep_line = "-" * len(header_line)

	print(header_line)
	print(sep_line)
	for r in rows:
		print(fmt_row(r))



def write_csv(rows: List[Dict], path: str):
	fieldnames = [
		"id",
		"type",
		"name",
		"start_date_local",
		"location_city",
		"location_state",
		"location_country",
		"distance_m",
		"moving_time_s",
		"elapsed_time_s",
		"average_speed_mps",
	]
	with open(path, "w", encoding="utf-8", newline="") as f:
		writer = csv.DictWriter(f, fieldnames=fieldnames)
		writer.writeheader()
		for r in rows:
			writer.writerow(
				{
					"id": r.get("id"),
					"type": r.get("type"),
					"name": r.get("name"),
					"start_date_local": r.get("start_date_local"),
					"location_city": r.get("location_city"),
					"location_state": r.get("location_state"),
					"location_country": r.get("location_country"),
					"distance_m": r.get("distance_m"),
					"moving_time_s": r.get("moving_time_s"),
					"elapsed_time_s": r.get("elapsed_time_s"),
					"average_speed_mps": r.get("average_speed_mps"),
				}
			)


def parse_date_to_epoch(value: Optional[str]) -> Optional[int]:
	if not value:
		return None
	try:
		# Accept YYYY-MM-DD
		d = dt.datetime.strptime(value, "%Y-%m-%d")
		return int(d.replace(tzinfo=dt.timezone.utc).timestamp())
	except ValueError:
		raise argparse.ArgumentTypeError("Date must be in YYYY-MM-DD format")


def main(argv: Optional[Iterable[str]] = None) -> int:
	parser = argparse.ArgumentParser(description="List all Strava activities for the authenticated athlete.")
	parser.add_argument("--csv", help="Optional path to write CSV output.")
	parser.add_argument("--after", type=str, default=None, help="Only activities after this date (YYYY-MM-DD)")
	parser.add_argument("--before", type=str, default=None, help="Only activities before this date (YYYY-MM-DD)")
	parser.add_argument("--per-page", type=int, default=200, help="Results per page (max 200)")
	parser.add_argument("--max-pages", type=int, default=None, help="Max pages to fetch (for testing)")
	parser.add_argument("--test", action="store_true", help="Fetch only a small sample (~15 activities) for quick testing.")
	parser.add_argument("--types", type=str, default="Walk,Hike", help="Comma-separated activity types to include (e.g., 'Walk,Hike'). Use '*' for all.")

	# OAuth helper options
	parser.add_argument("--print-auth-url", action="store_true", help="Print an OAuth URL to grant access and exit.")
	parser.add_argument("--scope", type=str, default="read,activity:read_all", help="Scopes to request when printing auth URL.")
	parser.add_argument("--redirect-uri", type=str, default="http://localhost/exchange_token", help="Redirect URI configured in your Strava app.")
	parser.add_argument("--exchange-code", type=str, default=None, help="Exchange an authorization CODE for tokens, then exit.")
	parser.add_argument("--write-secrets", action="store_true", help="When exchanging code, write strava_secrets.json next to this script.")

	# Location enrichment options
	parser.add_argument("--location-details", action="store_true", help="Fetch detailed activity data to fill city/state/country (extra API calls).")
	parser.add_argument("--details-max", type=int, default=None, help="Max number of detail calls to make (limits extra API usage).")
	parser.add_argument("--details-retries", type=int, default=3, help="Retries per detail call on transient errors.")
	parser.add_argument("--details-timeout", type=int, default=30, help="Timeout in seconds per detail call.")
	parser.add_argument("--detail-sleep-ms", type=int, default=0, help="Optional sleep in milliseconds between detail calls to be gentle on API.")
	parser.add_argument("--progress", action="store_true", help="Print progress while fetching details.")
	parser.add_argument("--progress-bar", action="store_true", help="Show a single-line progress bar while processing.")

	# Reverse geocoding options
	parser.add_argument("--geocode", action="store_true", help="Reverse geocode missing city/state/country using OSM Nominatim (external network).")
	parser.add_argument("--geocode-sleep-ms", type=int, default=0, help="Sleep between geocode calls (ms) to be polite.")
	# Download XML/TCX/GPX files structured by location
	parser.add_argument("--download", action="store_true", help="Download each activity export (GPX/TCX/original).")
	parser.add_argument("--download-format", choices=["gpx", "tcx", "original"], default="gpx", help="Export format for downloaded activities.")
	parser.add_argument("--download-dir", default="downloaded_activities", help="Base directory for organized activity downloads.")
	parser.add_argument("--download-quiet", action="store_true", help="Silence per-file download logs to keep output tidy.")
	parser.add_argument("--retro-convert-fit", action="store_true", help="Convert existing downloaded GPX files into minimal FIT format for Garmin import.")
	parser.add_argument("--retro-convert-fit-dir", default=None, help="Destination folder for converted FIT files (defaults to <root>/fit_converted).")
	parser.add_argument("--assume-naive-offset", default=None, help="Offset like +04:00 or -05:00 to apply to GPX timestamps that lack a timezone.")
	# Back-compat flags from README examples
	parser.add_argument("--repair-gpx-times", action="store_true", help="Deprecated: no-op; naive time handling is automatic. Use --tz-offset.")
	parser.add_argument("--tz-offset", default=None, help="Alias for --assume-naive-offset (e.g., -04:00).")
	parser.add_argument("--gpx-to-fit", nargs=2, metavar=("GPX_PATH","FIT_PATH"), default=None, help="Convert a single GPX file to FIT at the given path.")
	parser.add_argument("--force-tz-offset-for-all", action="store_true", help="Apply the --tz-offset/--assume-naive-offset to all timestamps, even if they already have a timezone.")
	parser.add_argument("--local-tz", default=None, help="IANA timezone name (e.g., America/Detroit) for DST-aware handling of naive timestamps.")
	parser.add_argument("--force-local-tz-for-all", action="store_true", help="Reinterpret all timestamps using the --local-tz zone (DST-aware), ignoring any existing offsets in GPX.")
	parser.add_argument("--validate-fit", type=str, default=None, help="Validate a FIT file or directory tree of FIT files.")
	parser.add_argument(
		"--rescan-unknown",
		action="store_true",
		help="Scan unknown_* folders and reorganize cached exports using local files; optionally pair with --geocode. No Strava API calls are made."
	)

	args = parser.parse_args(list(argv) if argv is not None else None)
	# Normalize legacy flags
	if args.tz_offset and not args.assume_naive_offset:
		args.assume_naive_offset = args.tz_offset

	if args.rescan_unknown:
		return rescan_unknown_locations(args)

	if args.retro_convert_fit:
		return run_retro_convert_fit(args)
	# Single-file conversion path
	if args.gpx_to_fit:
		gpx_path, fit_path = args.gpx_to_fit
		# Try to infer a type from parent folder name, default to Walk
		folder = os.path.basename(os.path.dirname(gpx_path))
		activity_type = folder.replace('_',' ') or 'Walk'
		ok = convert_gpx_file_to_fit(
			gpx_path, fit_path, activity_type,
			activity_name=None,
			prefer_generic_with_subsport=True,
			assume_naive_offset=args.assume_naive_offset,
			force_tz_offset_for_all=args.force_tz_offset_for_all,
			local_tz_name=args.local_tz,
			force_local_tz_for_all=args.force_local_tz_for_all,
		)
		print("Converted" if ok else "Failed", gpx_path, "->", fit_path)
		return 0 if ok else 1
	if args.validate_fit:
		path = args.validate_fit
		if os.path.isdir(path):
			for root, _dirs, files in os.walk(path):
				for name in files:
					if name.lower().endswith('.fit'):
						fp = os.path.join(root, name)
						res = validate_fit(fp)
						status = 'OK' if res.get('ok') else f"FAIL: {res.get('error')}"
						print(f"{fp}: {status}")
		else:
			res = validate_fit(path)
			print(json.dumps(res, indent=2))
		return 0

	# If user only wants to generate an auth URL, we may only need client_id.
	if args.print_auth_url:
		# Try to get client_id from env or secrets, else require manual input.
		client_id = os.environ.get("STRAVA_CLIENT_ID")
		if not client_id:
			secrets_path = os.path.join(os.path.dirname(__file__), "strava_secrets.json")
			if os.path.exists(secrets_path):
				try:
					with open(secrets_path, "r", encoding="utf-8") as f:
						data = json.load(f)
					client_id = str(data.get("client_id")) if data.get("client_id") else None
				except Exception:
					client_id = None
		if not client_id:
			print("Please provide STRAVA_CLIENT_ID via env var or create strava_secrets.json with client_id.")
			return 2
		url = build_auth_url(client_id, redirect_uri=args.redirect_uri, scope=args.scope)
		print("Open this URL to authorize the app and get a code:")
		print(url)
		print("\nAfter approval, copy the 'code' query parameter from the redirect URL and run:")
		print("  python .\\strava.py --exchange-code <code> [--write-secrets]")
		return 0

	# If exchanging a code, we need client credentials available.
	if args.exchange_code:
		try:
			creds = load_credentials()
		except StravaAuthError:
			# For exchange, at minimum client_id and client_secret must be present
			client_id = os.environ.get("STRAVA_CLIENT_ID")
			client_secret = os.environ.get("STRAVA_CLIENT_SECRET")
			if not (client_id and client_secret):
				print("To exchange a code, set STRAVA_CLIENT_ID and STRAVA_CLIENT_SECRET env vars or create strava_secrets.json with them.")
				return 2
			creds = {"client_id": client_id, "client_secret": client_secret, "refresh_token": ""}
		try:
			result = exchange_auth_code(creds["client_id"], creds["client_secret"], args.exchange_code)
		except (StravaAuthError, requests.RequestException) as e:
			print(f"Failed to exchange code: {e}")
			return 2
		print("Exchange successful. Save these credentials:")
		print(f"  client_id: {creds['client_id']}")
		print(f"  client_secret: <hidden>")
		print(f"  refresh_token: {result['refresh_token']}")
		if args.write_secrets:
			secrets_path = os.path.join(os.path.dirname(__file__), "strava_secrets.json")
			try:
				write_secrets_json(secrets_path, creds["client_id"], creds["client_secret"], result["refresh_token"])
				print(f"Wrote {secrets_path}")
			except Exception as e:
				print(f"Failed to write secrets file: {e}")
		return 0

	try:
		creds = load_credentials()
	except StravaAuthError as e:
		print(str(e))
		print()
		print("How to get credentials:")
		print("  1) Create a Strava API application at https://www.strava.com/settings/api")
		print("  2) Complete an OAuth flow to obtain a refresh token for your athlete.")
		print("  3) Set STRAVA_CLIENT_ID, STRAVA_CLIENT_SECRET, STRAVA_REFRESH_TOKEN in your environment,")
		print("     or create strava_secrets.json next to this script as documented in the header.")
		return 2

	after_epoch = parse_date_to_epoch(args.after)
	before_epoch = parse_date_to_epoch(args.before)

	try:
		access_token = get_access_token(
			creds["client_id"], creds["client_secret"], creds["refresh_token"]
		)
	except StravaAuthError as e:
		print(f"Authentication failed: {e}")
		return 2
	except requests.RequestException as e:
		print(f"Network error during token exchange: {e}")
		return 2

	rows_for_display: List[Dict] = []
	rows_for_csv: List[Dict] = []
	progress = SimpleProgress(args.progress_bar)
	details_calls = 0
	details_ok = 0
	geocode_cache: Dict[str, Dict[str, str]] = {}

	already_downloaded: Set[int] = set()
	total = 0
	matching = 0
	downloaded = 0
	cache_skipped = 0
	if args.download:
		os.makedirs(args.download_dir, exist_ok=True)
		already_downloaded = scan_downloaded_activity_ids(args.download_dir)
	try:
		# Determine type filter set
		type_filter = None
		if args.types and args.types.strip() not in ("*", "all", "ALL"):
			type_filter = {t.strip().lower() for t in args.types.split(',') if t.strip()}

		# Effective pagination; in --test cap to 15
		effective_per_page = max(1, min(200, int(args.per_page)))
		effective_max_pages = args.max_pages
		if args.test:
			effective_per_page = min(effective_per_page, 15)
			if effective_max_pages is None or effective_max_pages > 1:
				effective_max_pages = 1

		page = 1
		stop_pagination = False
		current_page_seen_matching = False
		current_page_has_new = False
		while True:
			batch = fetch_activity_page(
				access_token,
				page=page,
				per_page=effective_per_page,
				after=after_epoch,
				before=before_epoch,
			)
			if not batch:
				break
			if args.download:
				current_page_seen_matching = False
				current_page_has_new = False
			for act in batch:
				total += 1
				distance_m = act.get("distance") or 0.0
				moving_time_s = act.get("moving_time") or 0
				elapsed_time_s = act.get("elapsed_time") or 0
				name = act.get("name") or ""
				act_type = act.get("type") or ""
				start_date_local = act.get("start_date_local") or ""
				avg_speed = act.get("average_speed") or None
				activity_id_raw = act.get("id")
				if activity_id_raw is None:
					continue
				activity_id = int(activity_id_raw)

				# Filter non-matching types
				if type_filter is not None and act_type.lower() not in type_filter:
					continue
				if args.download:
					current_page_seen_matching = True
					cached_activity = activity_id in already_downloaded
				else:
					cached_activity = False
				if args.download and not cached_activity:
					current_page_has_new = True

				loc = get_city_state_country(act)
				# Optionally enrich via detail call
				if (
					not cached_activity and
					not (loc["city"] or loc["state"] or loc["country"]) and
					args.location_details
				):
						details_calls += 1
						try:
							progress_cb = (lambda msg: print(msg)) if args.progress else None
							detail = fetch_activity_details(
								access_token,
								int(act["id"]),
								timeout_s=max(1, int(args.details_timeout)),
								retries=max(1, int(args.details_retries)),
								progress_cb=progress_cb,
							)
							loc = get_city_state_country(detail)
							details_ok += 1
						except requests.RequestException:
							# Keep empty if details fail
							pass
						# Decrement details budget
						if args.details_max is not None:
							args.details_max -= 1
						# Optional small delay
						if args.detail_sleep_ms and args.detail_sleep_ms > 0:
							time.sleep(args.detail_sleep_ms / 1000.0)

				# If still missing and geocode enabled, try reverse geocoding (even if cached)
				if args.geocode and not (loc["city"] or loc["state"] or loc["country"]):
					latlng = act.get("start_latlng")
					if isinstance(latlng, (list, tuple)) and len(latlng) == 2 and all(isinstance(x, (int, float)) for x in latlng):
						lat = _round_coord(float(latlng[0]))
						lng = _round_coord(float(latlng[1]))
						key = f"{lat:.3f},{lng:.3f}"
						if key in geocode_cache:
							geo = geocode_cache[key]
						else:
							geo = reverse_geocode_nominatim(lat, lng)
							geocode_cache[key] = geo
							if args.geocode_sleep_ms:
								time.sleep(max(0, int(args.geocode_sleep_ms)) / 1000.0)
						# Only overwrite if useful
						if geo["city"] or geo["state"] or geo["country"]:
							loc = geo

				rows_for_display.append(
					{
						"id": act.get("id"),
						"type": act_type,
						"name": name,
						"date": to_local_date_str(start_date_local),
						"city": loc["city"],
						"state": loc["state"],
						"country": loc["country"],
						"distance_km": f"{(float(distance_m) / 1000.0):.2f}",
						"moving_time": fmt_duration(moving_time_s),
					}
				)

				rows_for_csv.append(
					{
						"id": act.get("id"),
						"type": act_type,
						"name": name,
						"start_date_local": start_date_local,
						"location_city": loc["city"],
						"location_state": loc["state"],
						"location_country": loc["country"],
						"distance_m": distance_m,
						"moving_time_s": moving_time_s,
						"elapsed_time_s": elapsed_time_s,
						"average_speed_mps": avg_speed,
					}
				)

				matching += 1

				if args.download:
					path = build_download_path(args.download_dir, act_type, loc, activity_id, args.download_format)
					has_file = os.path.exists(path)
					moved_file = False
					# If cached file exists under unknown_* tree, move it to the new resolved location
					if cached_activity and not has_file:
						unknown_path = build_download_path(
							args.download_dir,
							act_type,
							{"city": "", "state": "", "country": ""},
							activity_id,
							args.download_format,
						)
						if os.path.exists(unknown_path):
							os.makedirs(os.path.dirname(path), exist_ok=True)
							try:
								os.replace(unknown_path, path)
								moved_file = True
								has_file = True
								if not args.download_quiet:
									print(f"Moved cached activity {activity_id} -> {path}")
							except Exception as move_err:
								if not args.download_quiet:
									print(f"Failed to move cached activity {activity_id} from {unknown_path} to {path}: {move_err}")
					# If still cached but file not at target, drop from cache to trigger (re)download at correct location
					if cached_activity and not has_file:
						already_downloaded.discard(activity_id)
						cached_activity = False
					if cached_activity and has_file:
						cache_skipped += 1
						if not moved_file and not args.download_quiet:
							print(f"Skipping activity {activity_id}, already downloaded -> {path}")
					else:
						os.makedirs(os.path.dirname(path), exist_ok=True)
						fallback_source = "export"
						try:
							content = download_activity_file(access_token, activity_id, args.download_format)
						except requests.HTTPError as e:
							status = getattr(e.response, "status_code", None)
							if status == 403 and args.download_format == "gpx":
								try:
									streams = fetch_activity_streams(access_token, activity_id)
									content = build_gpx_from_streams(act, streams)
									fallback_source = "stream-based GPX"
								except (requests.RequestException, ValueError) as fallback_err:
									print(f"Failed to download activity {activity_id}: {fallback_err}")
									continue
							else:
								print(f"Failed to download activity {activity_id}: {e}")
								continue
						# Write downloaded content
						with open(path, "wb") as f:
							f.write(content)
							downloaded += 1
							already_downloaded.add(activity_id)
							if not args.download_quiet:
								print(f"Downloaded activity {activity_id} ({fallback_source}) -> {path}")

				# In test mode, stop after ~15 matching activities
				if args.test and matching >= 15:
					progress.done()
					stop_pagination = True
					break

				# Update simple progress bar after each activity processed
				progress.update(total, details_calls, details_ok, downloaded)
			if stop_pagination:
				break
			page += 1
			if effective_max_pages is not None and page > effective_max_pages:
				break
			continue
			break
	except requests.HTTPError as e:
		status = getattr(e.response, "status_code", None)
		print(f"HTTP error fetching activities: {e}")
		if status == 401:
			print("\nTips to fix 401 Unauthorized:")
			print("  - Ensure your refresh token belongs to this client_id and hasn't been revoked.")
			print("  - Re-authorize with proper scopes (at least 'activity:read'; include 'activity:read_all' to see private activities).")
			print("  - To re-authorize, run: python .\\strava.py --print-auth-url and follow the steps, then --exchange-code.")
		return 2
	except requests.RequestException as e:
		print(f"Network error fetching activities: {e}")
		return 2

	if rows_for_display:
		# finalize progress bar line
		progress.done()
		print_table(rows_for_display)
		print(f"\nFetched {matching} matching activities (queried {total} total).")
		if args.download:
			print(f"Downloaded {downloaded} activity exports into {args.download_dir}")
			if cache_skipped:
				print(f"Skipped {cache_skipped} downloads because files already existed.")
	else:
		print("No activities found with the given filters.")

	if args.csv:
		try:
			write_csv(rows_for_csv, args.csv)
			print(f"CSV written to {args.csv}")
		except Exception as e:
			print(f"Failed to write CSV: {e}")
			return 2

	return 0


if __name__ == "__main__":
	sys.exit(main())
