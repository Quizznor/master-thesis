/**
   \file
   Properties of nuclei

   \author Lukas Nellen
   \version $Id$
   \date 01 Jun 2005
*/

static const char CVSId[] =
  "$Id$";

#include <NucleusProperties.h>


namespace utl {

  NucleusProperties::NucleusProperties(const int theType) :
    fType(theType)
  {
  }


  int
  NucleusProperties::GetType()
    const
  {
    return fType;
  }


  int
  NucleusProperties::TypeCode(const unsigned int theCharge,
                              const unsigned int theAtomicNumber)
  {
    return
      kNucleusBase
      + kChargeFactor       * theCharge
      + kAtomicNumberFactor * theAtomicNumber;
  }


  /// Check if type code is a valid nucleus
  bool
  NucleusProperties::IsNucleus(const int theType)
  {
    return kNucleusBase < theType && theType < kNucleusMax;
  }

} // namespace utl


// Configure (x)emacs for this file ...
// Local Variables:
// mode: c++
// compile-command: "make -C .. -k"
// End:
