import urllib.request
import hashlib


def checksum(filename, md5):
    hash_md5 = hashlib.md5()
    with open(filename, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def download(url, filename, md5=None):
    data = urllib.request.urlopen(url).read()
    with open(filename, 'wb') as fout:
        fout.write(data)
    if not md5 is None:
        local_md5 = checksum(filename, md5)
        if local_md5 == md5:
            print(f"{filename}\tMD5 hash check passed")
        else:
            print(f"{filename}\tMD5 hash check NOT passed, pls re-download")
    else:
        print(f"{filename}\t not checking md5")

if __name__ == '__main__':
    # ESOL data
    download(
        'https://cloud.tsinghua.edu.cn/f/2cc3b125053a4275b6a2/?dl=1',
        'data/ESOL-solubility.csv',
        'ac1580ec494ad7a0f6f040f9afce96cf'
        )
    download(
        'https://cloud.tsinghua.edu.cn/f/d3460ae6efc747a8802b/?dl=1',
        'data/ESOL-solubility-readme.txt',
        'a0cfbfb4959ebf1f67b0685a5ef9fd9d'
        )
