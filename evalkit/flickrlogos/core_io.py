# -*- coding: utf-8 -*-
"""
Convenient I/O functions.

Author:   Stefan Romberg, stefan.romberg@informatik.uni-augsburg.de

Note: Script was developed/tested on Windows with Python 2.7

$Date: 2013-11-18 13:07:55 +0100 (Mo, 18 Nov 2013) $
$Rev: 7627 $$Date: 2013-11-18 13:07:55 +0100 (Mo, 18 Nov 2013) $
$HeadURL: https://137.250.173.47:8443/svn/romberg/trunk/romberg/research/FlickrLogos-32_SDK/FlickrLogos-32_SDK-1.0.4/scripts/flickrlogos/core_io.py $
$Id: core_io.py 7627 2013-11-18 12:07:55Z romberg $
"""
import sys, random
import re
import string
import zlib
from collections import defaultdict

if sys.version_info >= (3,0,0):
    from pickle import dump as pickle_dump
    from pickle import load as pickle_load
else:
    from cPickle import dump as pickle_dump
    from cPickle import load as pickle_load

import os
from os.path import exists, basename, dirname, join, isdir, normpath, abspath, split, sep
from os import makedirs, listdir

#===============================================================================
# helper classes
#===============================================================================

class Tee(object):
    """Simulates the behaviour of the unix program 'tee' to write output
    both to stdout *and* a file.
    """

    def __init__(self, name, mode="w"):
        self.file = None
        if name is not None and name != "-":
            print("Tee: Mirring stdout to file '"+name+"'")
            self.file = open(name, mode)
        self.stdout = sys.stdout
        sys.stdout = self

    def __del__(self):
        if self.file is not None:
            self.file.close()
        sys.stdout = self.stdout

    def write(self, data):
        if self.file is not None:
            self.file.write(data)
            #self.file.flush()
        self.stdout.write(data)

#===============================================================================
# helper methods
#===============================================================================

def filename(x):
    """Returns the filename without the directory part including extension."""
    return split(x)[1]

def icount(it):
    """Computes the length of some data given an iterator.
    Note: It consumes the iterator.
    """
    for size,_ in enumerate(it):
        pass
    return size+1

def msplit(s, delimiters=";,\t", strip=True, remove_empty_tokens=False, strip_linebreaks=True):
    """Splits the given string by any of the given delimiters.
    More sophisticated version of string.split() aka "multisplit".

    Usage examples:

    >>> msplit("abcd")
    ['abcd']
    >>> msplit("a,b,c,d")
    ['a', 'b', 'c', 'd']
    >>> msplit("a\\tb,c,d")
    ['a', 'b', 'c', 'd']
    >>> msplit("a\\tb,c;d e")
    ['a', 'b', 'c', 'd e']

    The parameter delimiters denotes *all* delimiters that are used to split
    the string into separate tokens. Delimiters *must be* single characters.
    Note: By default msplit() does not split the string at spaces.

    >>> msplit("a\\tb,c;d e", delimiters=";,\\t ")
    ['a', 'b', 'c', 'd', 'e']
    >>> msplit("a\\tb,c;d e", delimiters=";")
    ['a\\tb,c', 'd e']

    If strip is True (default) then split tokens will further be stripped
    of leading and trailing whitespaces.
    Examples:

    >>> msplit(" a,     b    , c  ", strip=True)
    ['a', 'b', 'c']
    >>> msplit(" a,     b    , c  ", strip=False)
    [' a', '     b    ', ' c  ']

    Note that if argument delimiter contains " " (space) as delimiter argument
    strip has no effect when set to True. Whitespaces are stripped from tokens
    *after* the original string has been split at delimiters. That means:

    >>> msplit("a b c ", delimiters=" ", strip=True)
    ['a', 'b', 'c', '']

    If strip_linebreaks is True (default) then line breaks will be removed
    before the string is split into tokens. This avoids trailing empty tokens:
    Examples:

    Note: strip=True swallows line breaks and so the last token will be empty:

    >>> msplit("a b c d \\n", delimiters=" ", strip=True, strip_linebreaks=True)
    ['a', 'b', 'c', 'd', '']
    >>> msplit("a b c d \\n", delimiters=" ", strip=True, strip_linebreaks=False)
    ['a', 'b', 'c', 'd', '']

    Note: strip=False will preserve the trailing line break as extra token:

    >>> msplit("a b c d \\n", delimiters=" ", strip=False, strip_linebreaks=True)
    ['a', 'b', 'c', 'd', '']
    >>> msplit("a b c d \\n", delimiters=" ", strip=False, strip_linebreaks=False)
    ['a', 'b', 'c', 'd', '\\n']

    If remove_empty_tokens is set to True then empty tokens are removed before
    the list of tokens is returned. By default remove_empty_tokens is False.
    Examples:

    >>> msplit("a,,b", remove_empty_tokens=True)
    ['a', 'b']
    >>> msplit("a,,b,", remove_empty_tokens=True)
    ['a', 'b']
    >>> msplit("", remove_empty_tokens=True)
    []
    >>> msplit(",", remove_empty_tokens=True)
    []
    >>> msplit(",,,", remove_empty_tokens=True)
    []

    >>> msplit("a,,b", remove_empty_tokens=False)
    ['a', '', 'b']
    >>> msplit("a,,b,", remove_empty_tokens=False)
    ['a', '', 'b', '']
    >>> msplit("", remove_empty_tokens=False)
    []
    >>> msplit(",", remove_empty_tokens=False)
    ['', '']

    Degenerated cases:

    >>> msplit("")
    []
    >>> msplit(",,,", remove_empty_tokens=False, strip_linebreaks=True)
    ['', '', '', '']
    >>> msplit(",,,", remove_empty_tokens=False, strip_linebreaks=False)
    ['', '', '', '']
    >>> msplit(",,,", remove_empty_tokens=True)
    []
    >>> msplit(None)
    Traceback (most recent call last):
      File "C:\EPD-6.2\lib\doctest.py", line 1248, in __run
        compileflags, 1) in test.globs
      File "<doctest __main__.msplit[27]>", line 1, in <module>
        msplit(5)
      File "F:\research\python\csv_scripts\csv_convert2cvectorfile.py", line 139, in msplit
        raise TypeError("msplit() expects string as first argument.")
    TypeError: msplit() expects string as first argument.
    >>> msplit(5)
    Traceback (most recent call last):
      File "C:\EPD-6.2\lib\doctest.py", line 1248, in __run
        compileflags, 1) in test.globs
      File "<doctest __main__.msplit[27]>", line 1, in <module>
        msplit(5)
      File "F:\research\python\csv_scripts\csv_convert2cvectorfile.py", line 139, in msplit
        raise TypeError("msplit() expects string as first argument.")
    TypeError: msplit() expects string as first argument.

    Autor: Stefan Romberg, stefan.romberg@informatik.uni-augsburg.de
    """
    if isinstance(s, bytes):
        s = s.decode('utf-8')
    if not isinstance(s, str):
        raise TypeError("msplit() expects string as first argument.")
    if s is None or len(s) == 0:
        return []

    if strip_linebreaks:
        if s[-2:] == "\r\n":
            s = s[0:len(s)-2]
        else:
            y = s[-1]
            if y == "\n" or y == "\r":
                s = s[0:len(s)-1]

    delim = delimiters[0]
    for d in delimiters[1:]:
        s = s.replace(d, delim)

    tokens = s.split(delim)

    if strip:
        tokens = [ t.strip() for t in tokens ]

    if remove_empty_tokens:
        tokens = [ t for t in tokens if len(t) > 0 ]

    return tokens

#===============================================================================
#
#===============================================================================

def csv_read(filename, delimiters=",\t;", grep=None, strip_tokens=True):
    """Reads a CSV file and returns a list of list holding each data value.

    * If *filename* is None or empty an exception is raised.
    * If *filename* does not exist None is returned.
    * If *filename* ends with ".zz" csv_read() tries to automatically read
      and decompress the file content with zlib.

    .. Seealso::
        :func:`csv_write`
        :func:`csv_iread`
    """
    if filename is None or filename == "":
        raise Exception("csv_read(): Cannot handle filename which is empty or None.")
    if not exists(filename):
        sys.stderr.write("csv_read(): File '"+filename+"' does not exist!")
        return None

    #TODO: exclude_columns
    #TODO: exclude_rows
    #TODO: convert numbers

    if filename.endswith(".zz"):
        # assume file content was compressed with zlib. (e.g. by csv_write)
        text  = read_compressed_file(filename)
        lines = text.splitlines()
        del text

        if delimiters is not None:
            data = []
            for line in lines:
                tokens = msplit(line, delimiters, strip=strip_tokens)
                data.append( tokens )

            if grep is not None:
                data = [ x for x in data if x.count(grep) > 0 ]

            return data
        else:
            if grep is not None:
                lines = [ x for x in lines if x.count(grep) > 0 ]

            return lines

    else:
        with open(filename, "rb") as f:
            data = []
            if delimiters is not None:
                if isinstance(delimiters, list):
                    for line in f:
                        tokens = msplit(line, delimiters, strip=strip_tokens)
                        data.append( tokens )

                elif isinstance(delimiters, str):
                    if strip_tokens:
                        for line in f:
                            tokens = msplit(line, delimiters, strip=strip_tokens)
                            data.append( tokens )
                    else:
                        for line in f:
                            data.append( line.split(delimiters) )
                else:
                    raise Exception("Cannot handle delimiter type: "+str(delimiters))
            else:
                data = [ line.strip() for line in f ]

            if grep is not None:
                data = [ x for x in data if x.count(grep) > 0 ]
            return data

def csv_write(list_of_list, filename, delimiter=",", compression=None, create_dirs=False):
    """Writes a list of lists to a CSV file.

    Example:

    >>> data_out = [ ["column1", '1', '11'], ["Test", '2', '22'] ]
    >>> csv_write(data_out, "testlist.txt")
    >>> os.path.exists("testlist.txt")
    True
    >>> data_in = csv_read("testlist.txt")
    >>> data_in == data_out
    True

    csv_write() can also write compressed CSV files. To enable compression
    set *compression* to a number in [0, 9]. The file will not be saved as
    "filename" but as "filename.zz".

    >>> data_out = [ ["column1", '1', '11'], ["Test", '2', '22'] ]
    >>> csv_write(data_out, "testlist2.txt", compression=9)
    >>> os.path.exists("testlist2.txt")
    False
    >>> os.path.exists("testlist2.txt.zz")
    True
    >>> data_in = csv_read("testlist2.txt.zz")
    >>> data_in == data_out
    True

    If *compression* is enabled then the list is converted
    to a CSV string and compressed to a file. During this operation the
    whole CSV table is kept as string in memory.

    .. Seealso::
        :func:`csv_read`
    """
    if create_dirs:
        outdir = dirname(filename)
        if not exists(outdir):
            makedirs(outdir)

    if compression is None:
        with open(filename, "wb") as f:
            for item in list_of_list:
                if isinstance(item, str):
                    line = item + '\n'
                else:
                    line = delimiter.join([ str(x) for x in item ]) + '\n'
                f.write( line.encode('utf-8') )

    else:
        if not isinstance(compression, int):
            compression = 9

        lines = []
        for item in list_of_list:
            if isinstance(item, str):
                line = item + '\n'
            else:
                line = delimiter.join([ str(x) for x in item ]) + '\n'
            lines.append( line )

        if not filename.endswith(".zz"):
            filename = filename + ".zz"
        
        write_compressed_file(filename, ''.join(lines), compression_level=compression)

#===============================================================================
# file compression
#===============================================================================

def write_compressed_file(filename, content, compression_level=9):
    """Writes the string *content* to the given file and compresses the data.

    >>> data_in = "Das ist \\nein Test\\n."
    >>> write_compressed_file("testfile2", data_in)

    >>> data_out = read_compressed_file("testfile2")
    >>> data_in == data_out
    True

    .. Seealso::
        :func:`read_compressed_file`
        :func:`csv_write`
    """
    assert isinstance(content, str), "Can only write strings."
    with open(filename, "wb") as f:
        f.write( zlib.compress(content.encode('utf-8'), compression_level) )

def read_compressed_file(filename):
    """Reads zlib compressed strings from a file.

    .. Seealso::
        :func:`write_compressed_file`
        :func:`csv_read`
    """
    if not exists(filename) or not os.path.isfile(filename):
        return None
    with open(filename, "rb") as f:
        return zlib.decompress( f.read() ).decode('utf-8')

#===============================================================================
# convenient wrappers for simple serialization of python objects
#===============================================================================

def savedump(data, filename, silent=False):
    """Serializes data to a file using the built-in cPickle module.

    .. Seealso::
        :func:`loaddump`
    """
    assert filename is not None and filename != "" and filename != "-", filename

    if not silent:
        print("savedump(): Saving data to '"+filename+"'..."),
    with open(filename, "wb") as f:
        pickle_dump(data, f, protocol=2)
    if not silent:
        print("Done")

def loaddump(filename, silent=False):
    """Unserializes data from a file that was written with
    the built-in cPickle module.

    .. Seealso::
        :func:`savedump`
    """
    assert filename is not None and filename != "" and filename != "-", filename

    if not exists(filename):
        if not silent:
            print("loaddump(): File '"+filename+"' does not exist. Returning None.")
        return None

    try:
        if not silent:
            print("loaddump(): Loading data from '"+filename+"'..."),
        with open(filename, "rb") as f:
            data = pickle_load(f)
        if not silent:
            print("Done")
        return data
    except Exception as ex:
        print("ERROR: loaddump(): Could not load data from '"+filename+"'.")
        print("       Passing exception to caller.")
        print(str(ex))
        raise ex

#===============================================================================
#
#===============================================================================

def exclude_dot_files(filelist):
    """Removes all files starting with '.' from a filelist."""
    return [ x for x in filelist if not x.startswith('.') ]

def is_image_file(filename):
    """Determines if filename indicates that the file is an image.

    Returns true if the filename ends with (case insensitive) one of the
    extensions '.jpg','.jpeg','.pgm','.png','.tif','.tiff','.gif','.bmp' and
    it does not start with '.'.
    (This hides the ._* files produced by Mac OS X aka "Apple Double files")
    """
    # assume no image file starts with '.': Hides ._* files produced by Mac OS X.
    if filename is None or filename == "" or filename.startswith('.'):
        return False

    x = filename.lower()
    return ( x.endswith(".jpg") or
             x.endswith(".jpeg") or
             x.endswith(".pgm") or
             x.endswith(".png") or
             x.endswith(".tiff") or
             x.endswith(".tif") or
             x.endswith(".gif") or
             x.endswith(".bmp") )

def clean_string(s):
    return re.sub('[%s]' % (string.punctuation+string.digits), '', s).lower()

def get_all_image_files(dir_path):
    """Returns a list of all image files within *dir_path*."""
    assert dir_path is not None

    files    = listdir(dir_path)
    images   = sorted(exclude_dot_files(files))
    imgFiles = [ ''.join( (dir_path, sep, x) ) for x in images if is_image_file(x) ]
    return imgFiles

def get_classes_by_filenames(image_directory):
    assert image_directory is not None

    oldcategory = ""
    images      = sorted(exclude_dot_files(listdir(image_directory)))

    classes = dict()
    for img in images:
        if not is_image_file(img):
            continue

        category = clean_string(basename(img)[0:-4])
        if category == oldcategory:
            continue

        oldcategory = category
        print("class: "+category)
        imgFiles = [ item for item in images if clean_string(basename(item)[0:-4]) == category ]
        classes[category] = imgFiles

    return classes

def get_classes_by_dirnames(base_directory):
    """Returns all image files below base_directory and their associated class.

    <base_directory>/class1/1.jpg
    <base_directory>/class1/2.jpg
    <base_directory>/class2/3.jpg
    <base_directory>/class2/4.jpg

    will yield a dictionary:
     "class1" -> [1.jpg, 2.jpg]
     "class2" -> [3.jpg, 4.jpg]
    """
    assert base_directory is not None
    directories = sorted(exclude_dot_files(listdir(base_directory)))

    classes = dict()
    for item in directories:
        dir_path = base_directory + sep + item
        if not isdir(dir_path):
            print("Warning: item '"+str(dir_path)+"' is not a directory. Skipping...")
            continue

        category = item
        print("class: "+category)

        imgFiles = get_all_image_files(dir_path)
        classes[category] = imgFiles

    return classes

def wildcards2regex(wildcard_pattern):
    """Converts a wildcard pattern such as "*.txt" to a regular expression."""
    p = wildcard_pattern
    p = p.replace("\\", "\\\\")

    p = p.replace(".", "\\.")
    p = p.replace("^", "\\^")
    p = p.replace("$", "\\$")
    p = p.replace("+", "\\+")
    p = p.replace("-", "\\-")
    p = p.replace("=", "\\=") # ?
    p = p.replace(",", "\\,") # ?
    p = p.replace("(", "\\(")
    p = p.replace(")", "\\)")
    p = p.replace("[", "\\[")
    p = p.replace("]", "\\]")
    p = p.replace("{", "\\{")
    p = p.replace("}", "\\}")
    p = p.replace("/", "\/}") # ?

    p = p.replace("*", ".*")
    p = p.replace("?", ".")
    return "^" + p + "$"

def grab_files(directory, wildcard_pattern=None, regex_pattern=None,
               topdown=True, followlinks=False):
    """Returns all files within directory AND its subdirectories that match
    the given pattern. Either wildcards or a regular expression can be used
    as pattern.

    If *followlinks* is set to True, symlinks are followed when traversing
    the directory. If *topdown* is set to False, subdirectories are traversed
    bottom-up.

    @author: Stefan Romberg
    """
    assert directory is not None and directory != ""
    assert ( ( wildcard_pattern is not None and regex_pattern is None ) or
             ( wildcard_pattern is None and regex_pattern is not None ) )

    # prepare regex engine
    if wildcard_pattern is not None:
        regex_pattern = wildcards2regex(normpath(wildcard_pattern))
    rec      = re.compile(regex_pattern)
    search   = rec.search

    # traverse directories
    matched_files = []
    for dir_path, dirnames, filenames in os.walk(directory,
                                                 topdown=topdown,
                                                 followlinks=followlinks):
        for filename in filenames:
            f_name = normpath(join(dir_path, filename))
            m      = search(f_name)
            if m is not None:
                assert m.start() == 0 and m.end() == len(f_name), (m.start(), m.end(), f_name)
                matched_files.append( f_name )

    return matched_files

def grab_files2(directory, suffix, topdown=True, followlinks=False):
    """Returns all files within directory AND its subdirectories that end
    with *suffix*.

    If *followlinks* is set to True, symlinks are followed when traversing
    the directory. If *topdown* is set to False, subdirectories are traversed
    bottom-up.

    @author: Stefan Romberg
    """
    assert directory is not None and directory != ""

    # traverse directories
    matched_files = []
    for dir_path, dirnames, filenames in os.walk(directory,
                                                 topdown=topdown,
                                                 followlinks=followlinks):

        print("grab_files2(): Processing dir: "+dir_path)
        for filename in filenames:
            if filename.endswith(suffix):
                matched_files.append( join(dir_path, filename) )

    return matched_files

#TODO: move method somewhere appropriate as it deals with a specific directory layout
def get_files_per_class_by_pattern(dir_path, classes, pattern):
    x = defaultdict(set)
    for c in classes:
        subdir = abspath(dir_path) + '/' + c + '/'
        files  = grab_files(normpath(subdir), pattern)
        x[c]   = files
        for f in files:
            assert exists(f), f
    return x

if __name__ == '__main__':
    #===========================================================================
    # TESTS
    #===========================================================================
    import doctest
    doctest.testmod()
    print("All doctests passed.")
