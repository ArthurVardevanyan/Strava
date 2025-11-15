#!/usr/bin/env python3

from __future__ import annotations

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
	"""Construct a GPX blob from Strava activity streams."""
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
	start_iso = activity.get("start_date_local") or activity.get("start_date")
	start_dt = _parse_iso_datetime(start_iso)
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
			time_el = ET.SubElement(trkpt, "time")
			time_el.text = pt_time.replace(microsecond=0).isoformat()
	return ET.tostring(track, encoding="utf-8", xml_declaration=True)
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
		"zoom": 10,
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
	parser.add_argument(
		"--rescan-unknown",
		action="store_true",
		help="Scan unknown_* folders and reorganize cached exports using local files; optionally pair with --geocode. No Strava API calls are made."
	)

	args = parser.parse_args(list(argv) if argv is not None else None)

	if args.rescan_unknown:
		return rescan_unknown_locations(args)

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
					# Respect details cap
					if args.details_max is None or args.details_max > 0:
						# Count an attempted details call
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

				# If still missing and geocode enabled, try reverse geocoding
				if not cached_activity and args.geocode and not (loc["city"] or loc["state"] or loc["country"]):
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
					if cached_activity and not has_file:
						already_downloaded.discard(activity_id)
						cached_activity = False
					if cached_activity and has_file:
						cache_skipped += 1
						if not args.download_quiet:
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
			if args.download and current_page_seen_matching and not current_page_has_new:
				print("All remaining matching activities already downloaded, stopping pagination.", flush=True)
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

