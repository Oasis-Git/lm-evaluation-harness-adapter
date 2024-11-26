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


class read_responses_processor:
    def __init__(self, file_path):
        assert file_path.endswith(".jsonl")
        self.file_path = file_path
        self.corpus = []
        for line in open(file_path, "r"):
            self.corpus.append(json.loads(line))

    def process(self, context):
        for response in self.corpus:
            if response["request"]["context"] == context:
                s = response["response"]
                context_length = len(context)
                return s[context_length:]
        print("No response found for context: ", context)
        return ""
