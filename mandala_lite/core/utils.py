from ..common_imports import *

def get_uid() -> str:
    return '{}'.format(binascii.hexlify(os.urandom(16)).decode('utf-8'))

class Hashing:
    """
    Helpers for hashing e.g. function inputs and call metadata.
    """
    @staticmethod
    def get_content_hash(obj:Any) -> str:
        """
        A deterministic hash function for python objects, one would hope
        """
        stream = io.BytesIO()
        joblib.dump(value=obj, filename=stream)
        stream.seek(0)
        m = hashlib.md5()
        m.update(str((stream.read())).encode())
        return m.hexdigest()