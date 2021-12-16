import pandas as pd
from data_templates import templates
from config import data_config

texts = []
queries = []

buildingenvironments = ['spare room', 'bedroom', 'bathroom', 'balcony', 'nursery', 'utility room', 'living room', 'lounge', 'family room', 'dining room',\
    'kitchen', 'garage', 'mud room', 'basement', 'games room', 'library', 'hall', 'shed', 'loft', 'attic', 'cellar', 'box room', 'landing', 'music room', \
    'office', 'pantry', 'guest room', 'toilet', 'restroom', "study room"]
devices = ['temperature', 'humidity', 'tv', 'clock', 'coffe maker', 'computer', 'dvd', 'radio', 'fan', 'printer', 'fridge', 'electrical  oven',\
    'microwave oven', 'heater', 'lamp', 'light', 'projector']

def data_generator(templates):
    for template in templates:
        text = template["text"]
        query =  template["query"]
        for device in devices:
            for buildingenvironment in buildingenvironments:
                text_i = text.replace("<A>", device)
                text_i = text_i.replace("<B>", buildingenvironment)
                texts.append(text_i)

                query_i = query.replace("<A>", device)
                query_i = query_i.replace("<B>", buildingenvironment)
                queries.append(query_i)
        data = pd.DataFrame()
        data["text"] = texts
        data["query"] = queries
    return data

if __name__ == '__main__':
    data = data_generator(templates)
    data.to_csv(data_config["data_dir"], index=False)
