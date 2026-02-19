import json
import os

MAPPING_FILE_1 = '/home/alexanderh/unsloth2/melody/filename_mapping.json'
MAPPING_FILE_2 = '/home/alexanderh/unsloth2/filename_mapping.json'
INPUT_JSONL = '/home/alexanderh/unsloth2/melody/train_data.jsonl'
OUTPUT_JSONL = '/home/alexanderh/unsloth2/melody/train_data_updated.jsonl'


def load_json(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: Could not find file {filepath}")
        exit(1)


def main():
    print("Loading mappings...")

    map1 = load_json(MAPPING_FILE_1)

    raw_map2 = load_json(MAPPING_FILE_2)

    peid_to_uuid_new = {v: k for k, v in raw_map2.items()}

    print(f"Mappings loaded. Processing {INPUT_JSONL}...")

    processed_count = 0
    updated_count = 0

    with open(INPUT_JSONL, 'r', encoding='utf-8') as infile, \
            open(OUTPUT_JSONL, 'w', encoding='utf-8') as outfile:

        for line_number, line in enumerate(infile, 1):
            if not line.strip():
                continue

            try:
                data = json.loads(line)

                if "messages" in data:
                    for message in data["messages"]:

                        if message.get("role") == "user" and "content" in message:
                            for item in message["content"]:
                                if item.get("type") == "image" and "image" in item:

                                    original_path = item["image"]

                                    dirname, filename = os.path.split(original_path)
                                    uuid_old, extension = os.path.splitext(filename)

                                    page_element_id = map1.get(uuid_old)

                                    if page_element_id:

                                        uuid_new = peid_to_uuid_new.get(page_element_id)

                                        if uuid_new:

                                            new_filename = f"{uuid_new}{extension}"
                                            new_path = os.path.join(dirname, new_filename).replace("\\", "/")

                                            item["image"] = new_path
                                            updated_count += 1
                                        else:
                                            print(
                                                f"Warning Line {line_number}: PageID '{page_element_id}' not found in Mapping 2.")
                                    else:

                                        pass

                outfile.write(json.dumps(data) + '\n')
                processed_count += 1

            except json.JSONDecodeError:
                print(f"Error: Invalid JSON on line {line_number}")

    print("-" * 30)
    print(f"Processing complete.")
    print(f"Lines processed: {processed_count}")
    print(f"Image paths updated: {updated_count}")
    print(f"Output saved to: {OUTPUT_JSONL}")


if __name__ == "__main__":
    main()
