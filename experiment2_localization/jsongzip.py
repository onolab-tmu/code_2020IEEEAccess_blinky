import gzip
import json

def dump(filename, data):
    '''
    Dump some data to json and gzip the resulting file.

    Parameters
    ----------
    filename: str
        The name of the file
    data: data structure
        The Python data structure to save
    '''

    json_str = json.dumps(data) + "\n"               # 2. string (i.e. JSON)
    json_bytes = json_str.encode('utf-8')            # 3. bytes (i.e. UTF-8)

    with gzip.GzipFile(filename, 'w') as fout:   # 4. gzip
        fout.write(json_bytes)                      

def load(filename):
    '''
    Load some data from a gzipped json file.

    Parameters
    ----------
    filename: str
        The name of the file
    '''

    with gzip.GzipFile(filename, 'r') as fin:    # 4. gzip
        json_bytes = fin.read()                      # 3. bytes (i.e. UTF-8)

    json_str = json_bytes.decode('utf-8')            # 2. string (i.e. JSON)
    data = json.loads(json_str)                      # 1. data

    return data
