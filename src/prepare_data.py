import argparse

from acl_anthology import Anthology
import pandas as pd


def prepare_abstracts_from_acl(threshold: int, keyword_string: str, output_path: str):
    # Instantiate the Anthology from the official repository
    anthology = Anthology.from_repo()

    selected_abstracts = []
    selected_ids = []

    for paper in anthology.papers():
        if keyword_string in str(paper.title).lower():
            if paper.abstract is None:
                continue
            #print(paper.full_id, paper.full_id, paper.title, paper.abstract)
            selected_abstracts.append(str(paper.abstract))
            selected_ids.append(paper.full_id)
            if len(selected_abstracts) == threshold:
                break

    df = pd.DataFrame({"ids": selected_ids, "abstracts": selected_abstracts})
    df.to_json(output_path, orient="records", lines=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=int, default=50)
    parser.add_argument("--keyword_string", type=str, default="semantic parsing")
    parser.add_argument("--output_path", type=str, default="data/selected_paper_abstracts.jsonl")
    args = parser.parse_args()
    prepare_abstracts_from_acl(args.threshold, args.keyword_string, args.output_path)
