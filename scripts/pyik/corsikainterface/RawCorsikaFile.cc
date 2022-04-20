/**
   \file
   Implement raw CORSIKA file

   \author Lukas Nellen
   \version $Id$
   \date 08 Dec 2003
*/


#include "RawCorsikaFile.h"
#include "CorsikaBlock.h"

#include <string>
#include <sstream>
#include <iostream>

using namespace std;
using std::string;
using namespace io::Corsika;

#define ERROR(mess) cerr << mess << endl;

RawFile::RawFile(const std::string& theName)
  : fDiskStream(theName.c_str()),
    fCurrentBlockNumber(0),
    fDiskUnthinnBlockBuffer(),
    fDiskThinnBlockBuffer(),
    fIndexInDiskBlock(0),
    fBlockBufferValid(false)
{
  if (!fDiskStream) {
    string msg = "Error opening Corsika file '" + theName + "'.\n";
    ERROR(msg);
    throw std::exception();
  }
}


void
RawFile::Open(const std::string& theName)
{
  if (fDiskStream.is_open()) {
    string msg = "Cannot open Corsika file '" + theName +
      "' - *this is already open";
    ERROR(msg);
    throw std::exception();
  }

  fDiskStream.open(theName.c_str());

  if (!fDiskStream) {
    string msg = "Error opening Corsika file '" + theName + "'.\n";
    ERROR(msg);
    throw std::exception();
  }

  fCurrentBlockNumber = 0;
  fIndexInDiskBlock   = 0;
  fBlockBufferValid = false;
}


void
RawFile::Close()
{
  if (fDiskStream.is_open())
    fDiskStream.close();
}


bool
RawFile::GetNextBlock(Block& theBlock)
{

  const bool thinned = true;

  if (!fBlockBufferValid) {
    if (!ReadDiskBlock(thinned)) {
      return false;
    }
  }

  theBlock = fDiskThinnBlockBuffer.fBlock[fIndexInDiskBlock];
  if (++fIndexInDiskBlock >= kBlocksInDiskBlock) {
    ++fCurrentBlockNumber;
    fIndexInDiskBlock = 0;
    fBlockBufferValid = false;
  }

  return true;
}


bool
RawFile::GetNextBlockUnthinned(BlockUnthinned& theBlock)
{

  const bool thinned = false;

  if (!fBlockBufferValid) {
    if (!ReadDiskBlock(thinned)) {
      return false;
    }
  }

  theBlock = fDiskUnthinnBlockBuffer.fBlock[fIndexInDiskBlock];

  if (++fIndexInDiskBlock >= kBlocksInDiskBlock) {
    ++fCurrentBlockNumber;
    fIndexInDiskBlock = 0;
    fBlockBufferValid = false;
  }

  return true;
}



bool
RawFile::ReadDiskBlock(const bool thinned)
{

  const unsigned int sizeThinned= sizeof(ThinnedDiskBlock);
  const unsigned int sizeUnthinned= sizeof(UnthinnedDiskBlock);

  const unsigned int size = thinned? sizeThinned : sizeUnthinned;

  if (fDiskStream.tellg() < 0){
    fDiskStream.clear();
  }

  if (!fDiskStream.seekg(fCurrentBlockNumber * size)){
    return false;
  }
  if (thinned){
    if (!fDiskStream.read(reinterpret_cast<char*>(&fDiskThinnBlockBuffer),
                          size))
      return false;
  }
  else{
    if (!fDiskStream.read(reinterpret_cast<char*>(&fDiskUnthinnBlockBuffer),
                          size))
      return false;
  }
  fBlockBufferValid = true;
  return true;
}


RawFile::PositionType
RawFile::GetNextPosition()
  const
{
  return fIndexInDiskBlock +
    kBlocksInDiskBlock * fCurrentBlockNumber;
}


void
RawFile::SeekTo(const PositionType thePosition)
{

  PositionType newBlockNumber = thePosition / kBlocksInDiskBlock;
  if (fCurrentBlockNumber != newBlockNumber) {
    fCurrentBlockNumber = newBlockNumber;
    fBlockBufferValid   = false;

  }
  fIndexInDiskBlock   = thePosition % kBlocksInDiskBlock;

}


bool
RawFile::IsOpen()
  const
{
  return fDiskStream.is_open();
}


// Configure (x)emacs for this file ...
// Local Variables:
// mode:c++
// compile-command: "make -C .. -k"
// End:
