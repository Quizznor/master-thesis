/* File: corsika.i */
%module corsika
%include "std_string.i"

%{
#include <iostream>
#include <iterator>
#define SWIG_FILE_WITH_INIT
#include "CorsikaUtilities.h"
#include "CorsikaParticle.h"
#include "CorsikaShowerFileParticleIterator.h"
#include "ShowerParticleList.h"
#include "ShowerParticleIterator.h"
#include "CorsikaShower.h"
#include "CorsikaShowerFile.h"
#include "CorsikaIOException.h"
%}

// This wraps the increment operator which is ignored
%inline %{
  void increment_particle_it(io::ShowerParticleIterator& it)
  {
    ++it;
  }
%}
%ignore io::ShowerParticleIterator::operator++;

%ignore CVSId;



%include "CorsikaUtilities.h"
%include "CorsikaParticle.h"
%include "CorsikaShowerFileParticleIterator.h"
%include "ShowerParticleList.h"
%include "ShowerParticleIterator.h"
%include "CorsikaShower.h"
%include "CorsikaShowerFile.h"
%include "CorsikaIOException.h"

// Generator so we can iterate over events in a Cosika file
%extend io::CorsikaShowerFile
{
  %pythoncode %{
     def events(self):
         self.GotoPosition(0)
         n = self.GetNEvents()
         for i in range(n):
             self.Read()
             yield self.GetCurrentShower()

               %}
};


// This translates the CorsikaIOException thrown inside next() into a python exception
%catches(io::CorsikaIOException) io::ShowerParticleIterator::next();

%extend io::ShowerParticleIterator {
  CorsikaParticle* current() {
    return &(*(*$self));
  }
  CorsikaParticle* next() {
    increment_particle_it(*$self);
    return &(*(*$self));
  }
 };

// A generator so we can iterate over particles in an event
// it stops when the CorsikaIOException is raised.
%extend io::CorsikaShower {
  %pythoncode %{
     def particles(self):
         iter = self.GetParticles().begin()
         yield iter.current()
         while True:
             try:
                 p = iter.next()
             except:
                 break
             yield p

  %}

 };

// instanciate the template
%template(CorsikaParticleList) io::ShowerParticleList< io::CorsikaShowerFileParticleIterator >;
