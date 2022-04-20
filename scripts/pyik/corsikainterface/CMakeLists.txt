cmake_minimum_required(VERSION 2.6)
project(corsika_reader CXX)

set (CMAKE_BUILD_TYPE Debug)
set (CMAKE_CXX_FLAGS_DEBUG "-Wall -ggdb3 -O0 -fno-inline")

set (CMAKE_INCLUDE_CURRENT_DIR TRUE)

set (Corsika_SOURCES
  CorsikaUtilities.h
  CorsikaUtilities.cc
  CorsikaParticle.h
  CorsikaParticle.cc
  CorsikaBlock.h
  CorsikaBlock.cc
  RawCorsikaFile.h
  RawCorsikaFile.cc
  NucleusProperties.h
  NucleusProperties.cc
  CorsikaShowerFileParticleIterator.h
  CorsikaShowerFileParticleIterator.cc
  CorsikaShowerFile.h
  CorsikaShowerFile.cc
  ShowerParticleIterator.h
  ShowerParticleIterator.cc
  ShowerParticleList.h
  CorsikaShower.h
  CorsikaShower.cc
  CorsikaIOException.h
)

add_library (CorsikaReader SHARED
  ${Corsika_SOURCES}
)

add_executable(read main.cc)

target_link_libraries(read
  CorsikaReader
  )

find_package (PythonLibs)
find_package (SWIG)

if (PYTHONLIBS_FOUND AND SWIG_FOUND)
  if (NOT AUGER_FIND_QUIETLY)
    message (STATUS "Python libs found")
    message (STATUS "PYTHON_LIBRARIES: ${PYTHON_LIBRARIES}")
    message (STATUS "PYTHON_INCLUDE_PATH: ${PYTHON_INCLUDE_PATH}")
    message (STATUS "PYTHON_DEBUG_LIBRARIES: ${PYTHON_DEBUG_LIBRARIES}")

    message (STATUS "Using SWIG-Python")
    message (STATUS "SWIG_DIR: ${SWIG_DIR}")
    message (STATUS "SWIG_EXECUTABLE: ${SWIG_EXECUTABLE}")
    message (STATUS "SWIG_VERSION: ${SWIG_VERSION}")
  endif (NOT AUGER_FIND_QUIETLY)

  include_directories (
    ${PYTHON_INCLUDE_PATH}
    )

  add_custom_command (OUTPUT corsika_wrap.cxx
    COMMAND ${SWIG_EXECUTABLE} -c++ -python ${CMAKE_CURRENT_SOURCE_DIR}/corsika.i
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/corsika.i ${Corsika_SOURCES}
    )

  add_library (corsika SHARED
    corsika_wrap.cxx
    )

  set_target_properties (corsika PROPERTIES PREFIX "_")
  target_link_libraries (corsika
    ${PYTHON_LIBRARIES}
    CorsikaReader
    )
else (PYTHONLIBS_FOUND AND SWIG_FOUND)
  message (STATUS "Either Python pr Swig not found")
endif (PYTHONLIBS_FOUND AND SWIG_FOUND)
