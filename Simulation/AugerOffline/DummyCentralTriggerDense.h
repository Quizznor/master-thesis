  /**
     \class DummyCentralTrigger

     \brief

     \author    Javier Gonzalez
     \date   08 Dec 2011
  */

#ifndef _DummyCentralTriggerDense_h_
#define _DummyCentralTriggerDense_h_

#include <fwk/VModule.h>
#include <evt/Event.h>
#include <utl/TimeInterval.h>

static const char CVSId_DummyCentralTriggerDense[] =
  "$Id: DummyCentralTrigger.h 23092 2013-03-20 15:02:38Z darko $";

namespace DummyCentralTriggerDenseNS {

  class DummyCentralTriggerDense : public fwk::VModule {

  public:

    DummyCentralTriggerDense();
    virtual ~DummyCentralTriggerDense();

    fwk::VModule::ResultFlag Init();
    fwk::VModule::ResultFlag Run(evt::Event & event);
    fwk::VModule::ResultFlag Finish();

  private:

    utl::TimeInterval fDefaultOffset;
    utl::TimeInterval fDefaultWindow;

    REGISTER_MODULE("DummyCentralTriggerDenseKG",DummyCentralTriggerDense) ;

  };

} // DummyCentralTrigger namespace

#endif

// Configure (x)emacs for this file ...
// Local Variables:
// mode:c++
// compile-command: "make  -k"
// End:
