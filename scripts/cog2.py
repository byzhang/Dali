#!/usr/bin/python
#!/usr/local/bin/python

import re
import sys

class CogCapture(object):
    def __init__(self):
        self.reset()

    def out(self, str):
        self._output.append(str)

    def outl(self, str):
        self._output.append(str + '\n')

    def msg(self, m):
        print("Message: ", m)

    def reset(self):
        self._output = []

    def get(self):
        return ''.join(self._output)

def  process(cog_filepath):
    with open(cog_filepath) as cog_f:
        cog_source = cog_f.read()

    cog_pattern = re.compile(r'(?s)/\*cog\s.*?\*/', re.MULTILINE)
    cog_inline_pattern = re.compile(r'(?s)/\*cog:.*?\*/', re.MULTILINE)

    cog_snippets = []
    for match in cog_pattern.finditer(cog_source):
        position = match.start()
        code = match.group()[6:-2]
        cog_snippets.append((position, code, match.group()))

    for match in cog_inline_pattern.finditer(cog_source):
        position = match.start()
        original_code = match.group()
        code = match.group()[6:-2]
        modified_code = code.replace(' ', "(\"\"\"", 1) + "\"\"\")"
        cog_snippets.append((position, modified_code, original_code))

    # make sure the code snippets get executed in order
    cog_snippets = sorted(cog_snippets)

    cog_outputs = []

    cog = CogCapture()
    variable_space = {'cog': cog}

    for _, modified_code, original_code in cog_snippets:
        exec(modified_code, variable_space)
        cog_source = cog_source.replace(original_code, cog.get())
        cog.reset()

    return cog_source

if __name__ == '__main__':
    res = process(sys.argv[1])
    print(res)
