import openai 
from openai import OpenAI
import pandas as pd
import os, requests, time, json, re
from io import StringIO
from bs4 import BeautifulSoup
import hashlib
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

endpoint = os.getenv('ENDPOINT')
api_key = os.getenv('OPENAI_API_KEY')
deployment_name = "gpt-4o"
client = OpenAI(
    base_url=endpoint,
    api_key=api_key
)

from supabase import create_client
SB_URL = os.getenv("SUPABASE_URL")
SB_KEY = os.getenv("SUPABASE_KEY")
if not (SB_URL and SB_KEY):
    raise RuntimeError("Set SUPABASE_URL and SUPABASE_KEY")
sb = create_client(SB_URL, SB_KEY)

SEASONS = ["2526", "2425", "2324"]              
CSV_URL = "https://www.football-data.co.uk/mmz4281/{season}/E0.csv"

def fetch_pl_csv():
    last_err = None
    for s in SEASONS:
        url = CSV_URL.format(season=s)
        try:
            df = pd.read_csv(url)
            if not df.empty:
                return df, url
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"Could not fetch PL CSV. Last error: {last_err}")

raw_df, source_url = fetch_pl_csv()
print("Odds source:", source_url)

# pick odds columns
def pick_cols(df, sets):
    for cols in sets:
        if all(c in df.columns for c in cols):
            return cols
    return None

odds_sets = [
    ("PSCH", "PSCD", "PSCA"),
    ("AvgH", "AvgD", "AvgA"),
    ("B365H", "B365D", "B365A"),
]
picked = pick_cols(raw_df, odds_sets)
if not picked:
    raise RuntimeError("No known odds columns found (PSCH/PSCD/PSCA or Avg*/B365*).")
home_col, draw_col, away_col = picked

if not all(c in raw_df.columns for c in ["HomeTeam", "AwayTeam"]):
    raise RuntimeError("CSV missing HomeTeam/AwayTeam.")

df = raw_df.rename(columns={
    "Date": "date",
    "HomeTeam": "home_team",
    "AwayTeam": "away_team",
    home_col: "home_odds",
    draw_col: "draw_odds",
    away_col: "away_odds",
})[["date","home_team","away_team","home_odds","draw_odds","away_odds"]].copy()

# light clean for consistent blob
df["match_date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce").dt.date.astype(str)
for c in ["home_odds","draw_odds","away_odds"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df = df.dropna(subset=["home_team","away_team","home_odds","draw_odds","away_odds"]).reset_index(drop=True)

# Use exactly 50 lines to match your expectation
df = df.head(50)
# ---- 2) RAW BLOB CREATION ----
lines = [
    f"{r['match_date']} | {r['home_team']} vs {r['away_team']} | H:{r['home_odds']} D:{r['draw_odds']} A:{r['away_odds']}"
    for _, r in df.iterrows()
]
blob = "\n".join(lines)
extracted_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

os.makedirs("data", exist_ok=True)
with open("data/raw_blob.txt", "w", encoding="utf-8") as f:
    f.write(blob)

# ---- 3) LLM STRUCTURING (in chunks) ----
def mk_id(match_date, home, away):
    base = f"{match_date}|{home}|{away}".lower()
    return hashlib.sha256(base.encode()).hexdigest()[:12]

def to_float(x):
    try:
        return float(str(x).replace(",", "").strip())
    except Exception:
        return None

all_records = []
CHUNK = 25  # 50 total -> two chunks, keeps tokens safe
total_lines = len(lines)

for start in range(0, total_lines, CHUNK):
    chunk_lines = lines[start:start+CHUNK]
    n = len(chunk_lines)
    chunk_blob = "\n".join(chunk_lines)

    system = "Return ONLY valid JSON (object). No prose. No markdown."
    user = f"""You will receive {n} lines. Each line has:
YYYY-MM-DD | HOME vs AWAY | H:<odd> D:<odd> A:<odd>

Return a JSON OBJECT exactly like: {{"records":[...]}} with EXACTLY {n} objects.
Each object MUST have these keys (and only these keys):
- match_date (string, YYYY-MM-DD)
- home_team (string)
- away_team (string)
- home_odds (number)
- draw_odds (number)
- away_odds (number)
- source_url (string) = "{source_url}"
- extracted_at (string) = "{extracted_at}"

MAPPING RULES (strict):
- Each input line maps to ONE object at the same index (1->1, 2->2, ...).
- Do not drop, merge, reorder, or add lines.
- Odds must be numbers (no text).

INPUT:
{chunk_blob}
"""

    resp = client.chat.completions.create(
        model=deployment_name,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        temperature=0,
        response_format={"type": "json_object"},
    )

    obj = json.loads(resp.choices[0].message.content)
    recs = obj.get("records", [])
    if not isinstance(recs, list) or len(recs) != n:
        raise RuntimeError(f"LLM returned {len(recs)} records for a chunk of {n} lines. Refuse to proceed.")

    # clean + coerce + compute IDs locally
    for r in recs:
        md = str(r.get("match_date") or "").strip()
        ht = str(r.get("home_team") or "").strip()
        at = str(r.get("away_team") or "").strip()
        ho = to_float(r.get("home_odds"))
        do = to_float(r.get("draw_odds"))
        ao = to_float(r.get("away_odds"))
        if not (md and ht and at and ho is not None and do is not None and ao is not None):
            continue
        all_records.append({
            "id": mk_id(md, ht, at),
            "match_date": md,
            "home_team": ht,
            "away_team": at,
            "home_odds": ho,
            "draw_odds": do,
            "away_odds": ao,
            "source_url": source_url,
            "extracted_at": extracted_at,
        })

with open("data/structured.json", "w", encoding="utf-8") as f:
    json.dump({"records": all_records}, f, ensure_ascii=False, indent=2)

print(f"LLM produced {len(all_records)} records. Sample:")
print(all_records[:3])

# ---- 4) UPSERT TO SUPABASE ----
BATCH = 500
for i in range(0, len(all_records), BATCH):
    sb.table("pl_odds").upsert(all_records[i:i+BATCH]).execute()

print(f"Upserted {len(all_records)} rows to Supabase -> pl_odds")
print("Files saved: data/raw_blob.txt, data/structured.json")