import os, requests
api_key = os.environ.get("OPENDOTA_API_KEY")

# Try parsedMatches endpoint — these are public matches that ARE parsed
r = requests.get("https://api.opendota.com/api/parsedMatches", params={"api_key": api_key})
print("parsedMatches:", r.status_code, len(r.json()), "matches")
if r.json():
    print("Sample:", r.json()[:3])

# Check one for leaver data
if r.json():
    mid = r.json()[0]["match_id"]
    r2 = requests.get(f"https://api.opendota.com/api/matches/{mid}", params={"api_key": api_key})
    d = r2.json()
    players = d.get("players", [])
    leavers = [p["leaver_status"] for p in players if p.get("leaver_status", 0) >= 2]
    has_gold = sum(1 for p in players if p.get("gold_t") and len(p.get("gold_t", [])) > 5)
    print(f"Match {mid}: {len(players)} players, {has_gold} with telemetry, {len(leavers)} leavers")