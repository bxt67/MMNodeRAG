import json
import os
DIR_PATH = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(DIR_PATH)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
image_dir = os.path.normpath(os.path.join(BASE_DIR, "InfoSeek", "wikipedia_images_sampled"))
decomposition_path = os.path.join(DIR_PATH, "data/decomposition.jsonl")
knowledge_base_path = os.path.join(BASE_DIR, "InfoSeek", "KnowledgeBase.jsonl")
processed_image_ids_path = os.path.join(DIR_PATH, "data/processed_image_ids.txt")

def is_image_file(filename):
    return os.path.splitext(filename)[1].lower() in IMAGE_EXTENSIONS

def extract_id(filename):
    return os.path.splitext(filename)[0]

def get_relevant_document(document_id):
    relevant_document = None
    with open(knowledge_base_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = json.loads(line)
            if line["wikidata_id"] == document_id:
                relevant_document = line
                break
    return relevant_document

def get_relevant_entities(document_id):
    relevant_entities = set()
    with open(decomposition_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = json.loads(line)
            doc_id = line["chunk_id"].split(":")[0]
            if doc_id == document_id:
                response = line['response']
                for s in response:
                    for e in s['entities']:
                        relevant_entities.add(e.strip().upper())
    return relevant_entities

def image_entity_matching(image_path, document, entities):
    return

def load_processed_image_ids():
    processed_ids = set()
    if os.path.exists(processed_image_ids_path):
        with open(processed_image_ids_path, 'r') as f:
            for line in f:
                processed_ids.add(line.strip())
    return processed_ids

def save_processed_image_id(image_id):
    with open(processed_image_ids_path, 'a') as f:
        f.write(image_id + '\n')

processed_image_ids = load_processed_image_ids()
image_entity_mapping = {}
for file in os.listdir(image_dir):
    if not is_image_file(file):
        continue
    file_id = extract_id(file)
    relevant_document = get_relevant_document(file_id)
    relevant_entities = get_relevant_entities(file_id)
    if relevant_document is None:
        save_processed_image_id(file)
        continue
    title = relevant_document["wikipedia_title"].upper()
    if title in relevant_entities:
        image_entity_mapping[file] = [title]
        save_processed_image_id(file)
    else:
        matched_entities = image_entity_matching(file, relevant_document, relevant_entities)
        if matched_entities:
            image_entity_mapping[file] = list(matched_entities)
            save_processed_image_id(file)

image_mapping_path = os.path.join(DIR_PATH, "data/image_entity_mapping.jsonl")
with open(image_mapping_path, 'w', encoding='utf-8') as f:
    for image_file, entities in image_entity_mapping.items():
        json_line = json.dumps({"image_file": image_file, "entities": entities})
        f.write(json_line + '\n')
print(f"Total images processed: {len([k for k, v in image_entity_mapping.items() if v])}")

#NOT FINISHED
