/**
   \file
   Exceptions in CORSIKA io

   \author Lukas Nellen
   \version $Id$
   \date 19 Nov 2003
*/

#ifndef _io_CorsikaIOException_h_
#define _io_CorsikaIOException_h_

static const char CVSId_io_CorsikaIOException[] =
  "$Id$";

#include <exception>
#include <string>

namespace io {

  /**
     \class CorsikaIOException CorsikaIOException.h "io/CorsikaIOException.h"
     \brief Base for exceptions in the CORSIKA reader
     \ingroup corsika exceptions
  */

  class CorsikaIOException : public std::exception {
  public:
    /// Construct CORISKA IO exception with message
    CorsikaIOException(std::string message) :
      fMessage(message) { }

    /// Virtual destructor for CorsikaIOException
    virtual ~CorsikaIOException() throw() { }

    const char* what() const throw() { return fMessage.c_str(); }

    /// Retrieve verbose exception name
    virtual std::string GetExceptionName() const
    { return "Corsika IO exception"; }
  private:
    std::string fMessage;

  }; // CorsikaIOException

} // io


#endif // _io_CorsikaIOException_h_

// Configure (x)emacs for this file ...
// Local Variables:
// mode: c++
// End:
