import os
import json


class write_request_processor:
    def __init__(self, file_path):
        assert file_path.endswith(".jsonl")
        if os.path.exists(file_path):
            os.remove(file_path)
        self.file_path = file_path

    def process(self, contexts, until, kwargs):
        with open(self.file_path, "a") as f:
            for context in contexts:
                request = dict()
                request["context"] = context
                request["until"] = until
                request.update(kwargs)
                f.write(json.dumps(request) + "\n")
