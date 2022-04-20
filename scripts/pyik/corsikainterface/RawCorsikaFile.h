/**
   \file
   Raw disk file as specified by the CORSIKA manual.

   \author Lukas Nellen
   \version $Id$
   \date 08 Dec 2003
*/

#ifndef _io_Corsika_RawFile_h_
#define _io_Corsika_RawFile_h_

#include "CorsikaBlock.h"
#include <string>
#include <fstream>

namespace io {

  namespace Corsika {

    /**
     \class RawFile RawCorsikaFile.h "Corsika/RawCorsikaFile.h"

     \brief Raw disk file.

     This class provides block-wise read access to a Corsika ground
     particles file on disk. Simple random access is supported.

     This class handles the grouping of individual blocks into a disk
     block with padding. It doesn't provide tools for unpacking the
     individual particles from a block.

     \author Lukas Nellen
     \date 08 Dec 2003
     \ingroup corsika
    */

    class RawFile {
    public:
      typedef unsigned long int PositionType;

      static const unsigned int kBlocksInDiskBlock = 21;

      /// Padding bytes at the beginning of a raw block
      static const unsigned int kPaddingBeginning  = 4;
      /// Padding bytes at the end of a raw block
      static const unsigned int kPaddingEnd        = 4;


      class ThinnedDiskBlock {
      private:
        /// initial padding - works also for size 0
        char fPaddingBeginning[kPaddingBeginning];
      public:
        Block  fBlock[kBlocksInDiskBlock];
      private:
        /// final padding - works also for size 0
        char fPaddingEnd[kPaddingEnd];

      };


      class UnthinnedDiskBlock {
      private:
        /// initial padding - works also for size 0
        char fPaddingBeginning[kPaddingBeginning];
      public:
        BlockUnthinned  fBlock[kBlocksInDiskBlock];
      private:
        /// final padding - works also for size 0
        char fPaddingEnd[kPaddingEnd];

      };



      RawFile()
        : fDiskStream(),
          fCurrentBlockNumber(0),
          fDiskUnthinnBlockBuffer(),
          fDiskThinnBlockBuffer(),
          fIndexInDiskBlock(0),
          fBlockBufferValid(false)
      { }

      /// Construct and open file
      RawFile(const std::string& theName);

      void Open(const std::string& theName);

      /// Close file (no-op for closed file).
      void Close();

      /// Read one block and advance
      bool GetNextBlock(Block& theBlock);

      /// Read one block and advance
      bool GetNextBlockUnthinned(BlockUnthinned& theBlock);

      /// Number of the block read by the next call to GetNextBlock
      PositionType GetNextPosition() const;

      /// Seek to a given block, the next block will be \a thePosition
      void SeekTo(const PositionType thePosition);

      /// Check if the file is open
      bool IsOpen() const;



    private:
      /// Read the block at the current position from disk
      bool ReadDiskBlock(const bool thinned);

      mutable std::ifstream fDiskStream;
      PositionType  fCurrentBlockNumber;

      UnthinnedDiskBlock fDiskUnthinnBlockBuffer;
      ThinnedDiskBlock   fDiskThinnBlockBuffer;


      unsigned int  fIndexInDiskBlock;
      bool          fBlockBufferValid;


    };

  } // Corsika
} // io


#endif // _io_Corsika_RawFile_h_

// Configure (x)emacs for this file ...
// Local Variables:
// mode:c++
// compile-command: "make -C .. -k"
// End:
