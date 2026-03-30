import json

processed_ids = set()
try:
    with open("data/processed_ids.txt", "r") as f:
        processed_ids = set(line.strip() for line in f)
except FileNotFoundError:
    pass

corpus_path = './InfoSeek/Wiki6M_ver_1_0.jsonl'
with open(corpus_path, "r", encoding="utf-8") as fin, \
     open("data/processed_ids.txt", "a") as idfile:

    for line in fin:
        record = json.loads(line)
        rid = str(record["id"])

        if rid in processed_ids:
            continue

        # --- PROCESS ---
        print(record)

        # --- MARK DONE ---
        idfile.write(rid + "\n")
        processed_ids.add(rid)