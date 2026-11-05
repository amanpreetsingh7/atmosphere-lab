"""Bounded HTTP-range ZIP access; never fall back to a whole-archive transfer."""
import io
import json
import re
import urllib.request
import zipfile
import zlib
from pathlib import Path, PurePosixPath
from .downloads import tls_context, digest_file


class RangeFile(io.RawIOBase):
    def __init__(self, url, size, opener=None):
        self.url, self.size, self.position = url, int(size), 0
        self.opener = opener or urllib.request.urlopen
        self.validator = None
        self.transferred = 0

    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.position

    def seek(self, offset, whence=0):
        value = offset if whence == 0 else self.position+offset if whence == 1 else self.size+offset if whence == 2 else -1
        if value < 0: raise ValueError('Invalid ZIP seek.')
        self.position = value
        return value

    def read(self, size=-1):
        count = self.size-self.position if size is None or size < 0 else min(size, self.size-self.position)
        if count <= 0: return b''
        if count > 32_000_000: raise ValueError('Refusing an oversized archive range request.')
        start, end = self.position, self.position+count-1
        headers = {'Range': 'bytes=%d-%d' % (start,end), 'Accept-Encoding':'identity', 'User-Agent':'Atmosphere-Lab/0.8'}
        if self.validator: headers['If-Range'] = self.validator
        request = urllib.request.Request(self.url, headers=headers)
        with self.opener(request, timeout=45, context=tls_context()) as response:
            expected = 'bytes %d-%d/%d' % (start,end,self.size)
            if response.status != 206 or response.headers.get('Content-Range') != expected:
                raise ValueError('The archive server did not honor the exact byte range. No full archive was downloaded; retry later.')
            if response.headers.get('Content-Encoding','identity').lower() != 'identity':
                raise ValueError('Encoded range responses are not supported.')
            tag = response.headers.get('ETag') or response.headers.get('Last-Modified')
            if self.validator and tag != self.validator: raise ValueError('Remote archive changed during installation.')
            if tag: self.validator = tag
            data = response.read(count+1)
        if len(data) != count: raise OSError('Incomplete archive range; retry the installation.')
        self.position += count
        self.transferred += count
        return data


def validate_member(info):
    name = PurePosixPath(info.filename)
    if name.is_absolute() or '..' in name.parts or '\\' in info.filename or info.is_dir() or '__MACOSX' in name.parts or name.name.startswith('._'):
        raise ValueError('Unsafe or non-spectrum archive member: '+info.filename)
    mode = info.external_attr >> 16
    if mode & 0o170000 == 0o120000: raise ValueError('Archive symlinks are not spectra.')
    if info.flag_bits & 1 or not 1 <= info.file_size <= 50_000_000 or info.compress_size > 32_000_000:
        raise ValueError('Requested spectrum is encrypted or exceeds the bounded member size.')


def selected_members(archive, names, directory, source):
    """CRC-check every selected member, preserve differing files, record provenance."""
    directory = Path(directory); directory.mkdir(parents=True,exist_ok=True)
    infos = [archive.getinfo(name) for name in names]
    if len(set(names)) != len(names) or len({Path(n).name for n in names}) != len(names):
        raise ValueError('Duplicate archive selections or output filenames.')
    for info in infos: validate_member(info)
    records=[]
    for index,info in enumerate(infos):
        target=directory/Path(info.filename).name
        print('Model %d/%d: %s' % (index+1,len(infos),target.name),flush=True)
        # Even a pre-existing member is checked against the official ZIP CRC;
        # the receipt also retains SHA256 for downstream provenance.
        if target.is_file():
            crc=0
            with target.open('rb') as stream:
                for chunk in iter(lambda:stream.read(1_048_576),b''): crc=zlib.crc32(chunk,crc)
            if target.stat().st_size != info.file_size or crc & 0xffffffff != info.CRC:
                raise ValueError('Existing model differs from the selected archive member. Move it aside before retrying: '+str(target))
        else:
            part=target.with_name(target.name+'.part')
            try:
                with archive.open(info) as stream,part.open('wb') as output:
                    total=0
                    while True:
                        chunk=stream.read(1_048_576)
                        if not chunk: break
                        total+=len(chunk)
                        if total>info.file_size: raise ValueError('Member exceeds its declared size.')
                        output.write(chunk)
                if part.stat().st_size != info.file_size: raise ValueError('Incomplete selected model.')
                part.replace(target)
            finally:
                if part.exists():part.unlink()
        records.append({'member':info.filename,'file':str(target),'bytes':info.file_size,
                        'crc32':'%08x'%info.CRC,'sha256':digest_file(target)})
    receipt=dict(source,checksum_scope='Selected members checked against ZIP CRC32; individual SHA256 recorded. Full archive MD5 was not checked.',members=records)
    part=directory/'installation_receipt.json.part'
    part.write_text(json.dumps(receipt,indent=2));part.replace(directory/'installation_receipt.json')
    return records
