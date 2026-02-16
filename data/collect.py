"""Data collection using /parsedMatches (parsed public matches with full telemetry)."""
import argparse, json, logging, os, time
from pathlib import Path
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
BASE_URL = "https://api.opendota.com/api"


class OpenDotaCollector:
    def __init__(self, api_key=None, output_dir="data/raw"):
        self.api_key = api_key or os.environ.get("OPENDOTA_API_KEY")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.min_interval = 0.05 if self.api_key else 1.1
        self.last_req = 0.0
        self.stats = {"attempts": 0, "http_fail": 0, "no_telemetry": 0, "saved": 0, "leavers_found": 0}

    def _rate_limit(self):
        elapsed = time.time() - self.last_req
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self.last_req = time.time()

    def _get(self, endpoint, params=None):
        url = f"{BASE_URL}{endpoint}"
        params = params or {}
        if self.api_key:
            params["api_key"] = self.api_key
        for attempt in range(3):
            self._rate_limit()
            try:
                r = self.session.get(url, params=params, timeout=30)
                if r.status_code == 200:
                    return r.json()
                elif r.status_code == 429:
                    wait = 60 * (attempt + 1)
                    logger.warning(f"Rate limited. Waiting {wait}s...")
                    time.sleep(wait)
                else:
                    return None
            except Exception as e:
                logger.warning(f"Error (attempt {attempt+1}): {e}")
                time.sleep(5 * (attempt + 1))
        return None

    def get_match_ids(self, less_than_match_id=None):
        params = {}
        if less_than_match_id:
            params["less_than_match_id"] = less_than_match_id
        matches = self._get("/parsedMatches", params)
        if not matches:
            return []
        return [m["match_id"] for m in matches]

    def get_match_details(self, match_id):
        self.stats["attempts"] += 1
        data = self._get(f"/matches/{match_id}")
        if not data:
            self.stats["http_fail"] += 1
            return None
        players = data.get("players", [])
        if len(players) != 10:
            self.stats["http_fail"] += 1
            return None
        duration = data.get("duration", 0)
        if duration < 600:
            return None
        has = sum(1 for p in players if p.get("gold_t") and len(p.get("gold_t", [])) > 5)
        if has < 5:
            self.stats["no_telemetry"] += 1
            return None
        # Track leaver stats
        leavers = sum(1 for p in players if p.get("leaver_status", 0) >= 2)
        if leavers > 0:
            self.stats["leavers_found"] += 1
        return data

    def collect(self, num_matches):
        ckpt_path = self.output_dir / "checkpoint.json"
        collected = set()
        last_id = None
        if ckpt_path.exists():
            with open(ckpt_path) as f:
                d = json.load(f)
                collected = set(d.get("collected_ids", []))
                last_id = d.get("last_match_id")
            logger.info(f"Resuming: {len(collected)} already collected")
        count = len(collected)
        logger.info(f"Target: {num_matches}. Have: {count}. Using /parsedMatches.")
        while count < num_matches:
            batch = self.get_match_ids(less_than_match_id=last_id)
            if not batch:
                logger.warning("No IDs returned. Waiting 10s...")
                time.sleep(10)
                continue
            last_id = min(batch)
            ok = 0
            for mid in batch:
                if count >= num_matches:
                    break
                if mid in collected:
                    continue
                det = self.get_match_details(mid)
                if det:
                    with open(self.output_dir / f"{mid}.json", "w") as f:
                        json.dump(det, f)
                    collected.add(mid)
                    count += 1
                    ok += 1
                    self.stats["saved"] = count
            logger.info(f"Batch: {ok}/{len(batch)} saved | Total: {count}/{num_matches} | {self.stats}")
            if count > 0 and count % 100 == 0:
                with open(ckpt_path, "w") as f:
                    json.dump({"collected_ids": list(collected), "last_match_id": last_id}, f)
        with open(ckpt_path, "w") as f:
            json.dump({"collected_ids": list(collected), "last_match_id": last_id}, f)
        logger.info(f"Done! {count} matches saved. Matches with leavers: {self.stats['leavers_found']}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num_matches", type=int, default=3000)
    p.add_argument("--output_dir", type=str, default="data/raw")
    p.add_argument("--api_key", type=str, default=None)
    args = p.parse_args()
    OpenDotaCollector(api_key=args.api_key, output_dir=args.output_dir).collect(args.num_matches)

if __name__ == "__main__":
    main()