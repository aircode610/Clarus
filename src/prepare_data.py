from acl_anthology import Anthology
import pandas as pd

# Instantiate the Anthology from the official repository
anthology = Anthology.from_repo()

threshold = 50
selected_abstracts = []
selected_ids = []

for paper in anthology.papers():
    if "semantic parsing" in str(paper.title).lower():
        if paper.abstract is None:
            continue
        #print(paper.full_id, paper.full_id, paper.title, paper.abstract)
        selected_abstracts.append(str(paper.abstract))
        selected_ids.append(paper.full_id)
        if len(selected_abstracts) == threshold:
            break

print(selected_abstracts[0], type(selected_abstracts[0]))

df = pd.DataFrame({"ids": selected_ids, "abstracts": selected_abstracts})
df.to_json("data/selected_paper_abstracts.json", orient="records", lines=True)
