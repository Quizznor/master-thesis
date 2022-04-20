# -*- coding: utf-8 -*-
"""New caching utilities specifically suited to keep and update caches of daily ADST production files. Uses hdf5 and pandas."""
import pandas as pd
import numpy as np


class ADSTCacher(object):
  """
  Cache ADST or other custom data into hdf5 files.

  Use cases:
  - caching of daily observer files if we don't want to reselect all files or wait for merged files,
  - cache a big simulation library of which the production is still on-going.
  """

  def __init__(self, paths, readfct,
               cachefn="./cache.hdf5",
               strict=False, pass_store=False,
               track_code=True,
               njobs=1, nchunk=1,
               verbose=False, test=False, **kwargs):
    """
    Main read-in part.

    Parameters
    ----------
    paths : string or list of strings
      Paths pointing to ADST or other input files.
      Unix wildcarts in strings are accepted.
    readfct : function
      Function used to process data indicated by paths.
      This function needs to be suited to read the data.
    cachefn : string, optional
      Filename of the cache to store data in.
    strict : boolean, opional (default: True)
      If True, will keep track of individual files via sha512 hashes in addition to filenames, mod dates and sizes.
      Beware that for this to work files might need to be read twice in order to calculate the hash value.
      This will slow down the caching and can be disabled.
    pass_store : boolean, optional (default: False)
      If True, the hdf5 store will be passed as a second argument to readfct.
      It will then be assumed that readfct will take care of appending to the store within itself.
    track_code : boolean, optional (default: True)
      Option to keep track of source code of readfct and rebuilt cache if deemed necessary.
      Beware that the tracking starts when track_code is enabled first.
    njobs : integer, optional (default: 1)
      If > 1, will read files in parallel using pmap.
    nchunks : integer, optional (default: 1)
      Will feed #nchunks input files together into each call of readfct
    verbose : boolean, optional (default: False)
      If True, will print more output for debugging.
    test : boolean, optional (default: False)
      Test mode: if True, will do all operations except the actual caching and processing of data
    kwargs:
      Additional keyword arguments are passed through to readfct
    Returns
    -------
    """
    import os
    import hashlib
    import inspect
    from glob import glob
    from collections import defaultdict
    from pyik.performance import pmap

    self.paths = paths
    self.readfct = readfct
    self.cachefn = cachefn

    # if paths is a string, we assume that it is either the path to one file
    # or to multiple files specified with unix wildcarts (*,?,...)
    if isinstance(paths, str):
      plist = glob(paths)
    else:
      pll = [glob(pi) for pi in paths]
      # flattens to 1d list if 2d
      plist = [pi for psub in pll for pi in psub]

    plist = sorted(plist)

    if len(plist) < 1:
      print "No input files specified! Exiting!"
      return

    print "List containing {0} input files is provided. Proceeding to read files.".format(len(plist))
    print "If this script fails and you are uncertain why, please consider to delete your cache file and start from scratch!"

    finfo = defaultdict(list)
    for f in plist:
      fsize = os.path.getsize(f)
      finfo["tmod"].append(os.path.getmtime(f))
      finfo["tcreate"].append(os.path.getctime(f))
      finfo["fsize"].append(fsize)
      finfo["path"].append(os.path.abspath(f))
      if strict:
        with open(f) as inp:
          # calculation of file hashs is slow and requires lots of ram for large files
          # go around this by just taking the filesize in bytes for very large files > 500 MB
          if fsize > 500 * 1024 * 1024:
            finfo["fhash"].append(fsize)
          else:
            finfo["fhash"].append(hashlib.sha512(inp.read()).hexdigest())
      else:
        finfo["fhash"].append(0)

    finfo = pd.DataFrame(finfo)

    if verbose:
      print finfo

    if not os.path.exists(cachefn):
      print "Cache {0:s} doesn't exist yet! A new cache will be created.".format(cachefn)
    store = pd.HDFStore(cachefn, "a", complib="blosc", complevel=9)
    self.store = store

    code_hash = hashlib.sha512(inspect.getsource(readfct)).hexdigest()

    if "/processed" in store.keys():
      processed = store.processed
      print "Cache contains data of {0} processed files.".format(len(processed))
      if verbose:
        print processed
    else:
      print "Cache does not contain data, will proceed to process all data."
      store.put("processed", pd.DataFrame(), format="table")
      processed = None

    toprocess = pd.DataFrame()
    reprocess = False

    if track_code:
      if "/code_hash" in store.keys():
        saved_hashs = store.code_hash.values
        if code_hash not in saved_hashs:
          print "Code tracking is active and your read function seems to have changed since the cache was created."
          print "You can choose to (r)ebuilt the cache (the previous cache is lost) or (i)gnore this (e.g. if you know that the change was irrelevant)."

          answers = ["r", "i"]
          answer = None
          while answer not in answers:
            answer = raw_input("Rebuilt (r) or Ignore (i): ")
            answer = answer.lower()

          if answer == "r":
            reprocess = True
            store.put("code_hash", pd.Series(code_hash), format="table")
          else:
            store.append("code_hash", pd.Series(code_hash))
      else:
        store.put("code_hash", pd.Series(code_hash), format="table")

    if not reprocess:

      for i, fn in enumerate(plist):

        finfoi = finfo[i:i + 1]

        if processed is not None:

          if not strict:
            sel = processed[processed.path == fn]
            if len(sel) == 0:
              print "File {0} is not found within the cache, will be processed.".format(fn)
              toprocess = toprocess.append(finfoi)
            elif len(sel) == 1:
              if sel.tmod.values[0] != finfoi.tmod.values[0]:
                print "File {0} was modified after cache was built, need to rebuild the cache!".format(fn)
                reprocess = True
              if sel.fsize.values[0] != finfoi.fsize.values[0]:
                print "File {0} had a different size when cache was built, need to rebuild the cache!".format(fn)
                reprocess = True
            elif len(sel) > 1:
              print "Several entries for file {0} are found in the cache! This is unexpected behaviour, please check!".format(fn)

          else:
            sel = processed[processed.fhash == finfoi.fhash.values[0]]
            if len(sel) == 0:
              print "File {0} is not found within the cache, will be processed.".format(fn)
              toprocess = toprocess.append(finfoi)
            elif len(sel) == 1:
              if not np.all(finfoi == sel):
                print "Cache contains data of file {0} but file information (size, timestamp, name) changed! Will overwrite file information!"
                processed.loc[processed.fhash == finfoi.fhash.values] = finfoi
                store.put("processed", processed, format="table")
            elif len(sel) > 1:
              print "Several entries for file {0} are found in the cache! This is unexpected behaviour, please check!".format(fn)

          if reprocess:
            break
        else:
          toprocess = toprocess.append(finfoi)

    # need to reprocess all files
    if reprocess:
      toprocess = finfo

    for i in range(toprocess.shape[0]):

      topi = toprocess.iloc[i]

      # obtain hashes for the files that need to be processed (also if strict option is off)
      # at this point, the files need to be read (transferred via network) anyway...
      if not strict:
        with open(topi.path) as inp:
          # see comment above concerning hashing and file sizes
          if os.path.getsize(topi.path) > 500 * 1024 * 1024:
            fhash = os.path.getsize(topi.path)
          else:
            fhash = hashlib.sha512(inp.read()).hexdigest()
          toprocess.loc[toprocess.path == topi.path, "fhash"] = fhash

      if ((i + 1) % (njobs * nchunk) == 0) or (i == toprocess.shape[0] - 1):

        n = i % (njobs * nchunk)

        topiall = toprocess[i - n:i + 1]
        pathsi = topiall.path.values

        nj = min(njobs, int((n + 1) * 1. / nchunk))
        nj = max(nj, 1)
        pathsi_chunks = np.array_split(pathsi, nj)

        if test:
          rfct = lambda x: 0
        else:
          if pass_store:
            def rfct(plist):
              return readfct(plist, store, **kwargs)
          else:
            def rfct(plist):
              return readfct(plist, **kwargs)

        # store needs to be opened (and closed) within readfct for parallelization
        if pass_store:
          store.close()

        if nj > 1:
          dfs_map = pmap(rfct, pathsi_chunks, chunksize=nj, numprocesses=nj)
        else:
          dfs_map = map(rfct, pathsi_chunks)

        if not store.is_open:
          store.open()

        if not test:

          if not pass_store:

            for df_map in dfs_map:

              # df_map is expected to be a dictionary with data frames as values
              for df_map_k, df_map_v in df_map.items():

                if ("/%s" % df_map_k) in store.keys():
                  store.append(df_map_k, df_map_v, format="table", data_columns=True)
                else:
                  store.put(df_map_k, df_map_v, format="table", data_columns=True)

          store.append("processed", topiall, format="table")

        print "Processed {0} of {1} new input files.".format(i + 1, len(toprocess))

      store.close()

  def __str__(self):
    print "This is a cache object!"

  def GetInputFileInformation(self):
    self.store.open()
    info = self.store.processed
    self.store.close()
    return info

  def GetInputFileList(self):
    self.store.open()
    fl = self.store.processed.path.values
    self.store.close()
    return fl
